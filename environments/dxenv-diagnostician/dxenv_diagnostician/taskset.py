"""dxenv-diagnostician: cost-aware diagnosis over synthetic patients.

The agent works a case one turn at a time: order a test (it costs budget and needs a
committed prediction of the result), prescribe, or end the episode with a probability
distribution over 149 conditions (or abstain). One reward per episode, a strictly proper
Brier score weighted by the true condition's severity, minus test and turn costs, plus
treatment and predict-then-verify terms.

The policy is STATELESS between turns, as in the original environment: every turn is a
fresh single-turn rollout whose prompt is rebuilt from the typed observation (which has no
field that could carry the label). Each turn is one trace; every trace of an episode
carries the episode's return.

Environment mechanics, observation, prompt, action parser and reward are imported
unchanged from the source repo (`dxenv`). Two things differ, both documented in the README:

1. Coherence fix. The source treatment term pays `0.5 * coherence` for prescriptions that
   fit the agent's OWN declared diagnosis, true or not, which a no-information policy farms
   for +0.46. Here that term is gated on the mass the report puts on the truth:
   `coherence * mass_on_truth`. Implemented as an adjustment on top of the source score,
   so every other term is the source's arithmetic; the source total is logged as the
   `source_reward` metric.
2. Output constraint. On ARC, vLLM's guided decoding enforced the source action schema
   (`dxenv.policy.decoding.action_json_schema`), so off-menu keys and invalid prediction
   words were unrepresentable. Here the same schema is sent as an OpenAI-style
   `response_format` (on by default), and the wire format is also spelled out in the
   system prompt for providers that ignore schemas. A reply that still does not parse
   ends the episode as `decode_failure`, which the source scores as a non-decision.
"""

import dataclasses
import re
from functools import cache
from typing import Any, Literal

import numpy as np
from dxenv.data.corpus import PatientRecord, generate_corpus
from dxenv.data.splits import rebuild_frozen_splits
from dxenv.data.taxonomy import load_taxonomy
from dxenv.env.actions import build_menu
from dxenv.env.catalog import CategoricalAnalyte, load_catalog
from dxenv.env.episode import DiagnosticEpisode, load_episode_config, sample_budget
from dxenv.env.obs_model import build_observation_model
from dxenv.policy.decoding import action_json_schema, parse_action
from dxenv.policy.prompt import SYSTEM_PROMPT as SOURCE_SYSTEM_PROMPT
from dxenv.policy.prompt import build_prompt
from dxenv.reward.engine import GroundTruth, RewardConfig, load_reward_config, score_trajectory

import verifiers.v1 as vf

Split = Literal["train", "eval", "holdout"]
_SPLIT_CODE = {"train": 0, "eval": 1, "holdout": 2}
BUDGET_SEED = 20260901
"""Per-patient budgets are drawn from p(B) with rng([BUDGET_SEED, split code, index]), so a
task's budget is fixed across runs and across the samples of a group."""


@cache
def _corpus() -> tuple[dict[str, PatientRecord], dict[str, tuple[str, ...]]]:
    """The source repo's frozen 20k-patient corpus and its train / eval / holdout split.

    Regenerated from the recorded provenance (seed 20260901) and checked against the
    committed eval hash by `rebuild_frozen_splits`, so this is THE eval split, not a
    look-alike. Takes ~12 s once per process.
    """
    records: list[PatientRecord] = []

    def generate(n: int, seed: int) -> list[PatientRecord]:
        records[:] = generate_corpus(n, seed=seed)
        return records

    splits = rebuild_frozen_splits(generate=generate)
    by_id = {r.patient_id: r for r in records}
    return by_id, {
        "train": splits.train,
        "eval": splits.eval,
        "holdout": splits.holdout_modules,
    }


@cache
def _world() -> dict[str, Any]:
    """Global objects every episode shares; none of them depends on a patient [I3]."""
    return {
        "episode_config": load_episode_config(),
        "reward_config": load_reward_config(),
        "taxonomy": load_taxonomy(),
        "catalog": load_catalog(),
        "menu": build_menu(),
        "model": build_observation_model(),
    }


def _categorical_tests() -> list[str]:
    w = _world()
    return sorted(
        a.key
        for a in w["menu"].test_actions()
        if isinstance(w["catalog"].analyte(w["catalog"].test(a.key).analytes[0]), CategoricalAnalyte)
    )


def response_format() -> dict[str, Any]:
    """The source grammar as an OpenAI-style structured-output schema.

    Identical to the schema vLLM enforced on ARC except two portability edits that do not
    change which objects are valid: the top-level `oneOf` becomes `anyOf` (the four variants
    are disjoint on the `kind` constant, so exactly one can match either way; several
    providers reject `oneOf`), and the `$schema` key is dropped."""
    schema = action_json_schema(build_menu(), load_taxonomy())
    schema.pop("$schema", None)
    schema["anyOf"] = schema.pop("oneOf")
    return {"type": "json_schema", "json_schema": {"name": "dxenv_action", "schema": schema}}


@cache
def system_prompt() -> str:
    """The source system prompt plus the wire format the source enforced with a grammar."""
    return f"""{SOURCE_SYSTEM_PROMPT}

ACTION FORMAT. Exactly one of these JSON objects, with no text before or after it:
  {{"kind":"order_test","reasoning":"...","test_key":"<test_key>","prediction":"low|normal|high"}}
  {{"kind":"prescribe","reasoning":"...","treatment_key":"<treatment_key>"}}
  {{"kind":"diagnose","reasoning":"...","diagnosis":[{{"condition":"<condition>","probability":"0.6"}}]}}
  {{"kind":"abstain","reasoning":"..."}}
- test_key, treatment_key and condition must be copied exactly from the lists in the case:
  the word before the parentheses, without the category heading. Only listed TESTS can be
  ordered; an individual lab value (a result name) is not a test.
- "prediction" must be exactly one of: low, normal, high, normal_categorical,
  abnormal_categorical. Use normal_categorical / abnormal_categorical for these tests,
  whose result is categorical: {", ".join(_categorical_tests())}.
  Use low / normal / high for every other test.
- A diagnosis names at most 16 conditions. Probability mass you leave unassigned is
  spread evenly over every condition you did not name.
- A reply that is not one valid JSON object, or that names anything not on the lists,
  ends the case immediately and is scored like abstaining."""


_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text: str) -> str:
    """Strip the wrappers API models put around a JSON object (a markdown fence, a
    <think> block, a sentence of preamble). Changes no field of the object, so it cannot
    change which action is taken; anything still unparseable is a decode failure."""
    t = _THINK.sub("", text or "").strip()
    m = _FENCE.search(t)
    if m:
        return m.group(1)
    if not t.startswith("{") and "{" in t and "}" in t:
        return t[t.index("{") : t.rindex("}") + 1]
    return t


def coherence_adjustment(trajectory: dict[str, Any], true_condition: str, cfg: RewardConfig) -> float:
    """The Hub change to the treatment term, as a delta on the source score.

    Source (dxenv/reward/treatment.py:161-177): coherence_scale * coherence, where coherence
    is the mean appropriateness of the prescriptions for the agent's declared argmax.
    Hub: coherence_scale * coherence * mass_on_truth. The difference is
    -coherence_scale * coherence * (1 - mass_on_truth). `declared` and the argmax are
    computed exactly as `score_trajectory` computes them, ties included.
    """
    prescribed: list[str] = []
    declared: dict[str, float] = {}
    for step in trajectory["steps"]:
        action = step["action"]
        if action["kind"] == "prescribe":
            prescribed.append(action["treatment_key"])
        elif action["kind"] == "diagnose":
            declared = {str(k): float(v) for k, v in action["distribution"].items()}
        elif action["kind"] == "abstain":
            declared = {}
    if not prescribed or not declared:
        return 0.0
    declared_dx = max(declared, key=lambda k: declared[k])
    coherence = sum(cfg.treatments.appropriateness(t, declared_dx) for t in prescribed) / len(
        prescribed
    )
    return -cfg.coherence_scale * coherence * (1.0 - declared.get(true_condition, 0.0))


def score(trajectory: dict[str, Any], record: PatientRecord) -> tuple[float, dict[str, Any]]:
    """(Hub reward, facts). Hub reward = source `score_trajectory` total + coherence fix."""
    w = _world()
    cfg = w["reward_config"]
    b = score_trajectory(
        trajectory,
        GroundTruth(record.condition, record.analytes, record.allergies),
        cfg,
        taxonomy=w["taxonomy"],
        catalog=w["catalog"],
        model=w["model"],
    )
    adj = coherence_adjustment(trajectory, record.condition, cfg)
    declared = next(
        (s["action"]["distribution"] for s in trajectory["steps"] if s["action"]["kind"] == "diagnose"),
        {},
    )
    return b.total + adj, {
        **b.as_dict(),
        "source_reward": b.total,
        "coherence_adjustment": adj,
        "treatment_hub": b.treatment + adj,
        "mass_on_truth": float(declared.get(record.condition, 0.0)),
        "top1_correct": bool(declared) and max(declared, key=lambda k: declared[k]) == record.condition,
    }


def new_episode(record: PatientRecord, seed: int, budget: float, max_turns: int) -> DiagnosticEpisode:
    w = _world()
    cfg = w["episode_config"]
    if max_turns != cfg.max_turns:
        cfg = dataclasses.replace(cfg, max_turns=max_turns)
    return DiagnosticEpisode(
        record, seed=seed, config=cfg, menu=w["menu"], catalog=w["catalog"], budget=budget
    )


def apply_reply(episode: DiagnosticEpisode, reply: str) -> str | None:
    """Parse one model reply and step the episode. Returns the error on a decode failure.

    A reply that does not parse, or that the environment rejects, ends the episode as
    `decode_failure`, exactly as the source rollout loop does (policy/rollout.py:285-294):
    the failed turn is not added to the trajectory and no action is invented for it.
    """
    w = _world()
    try:
        episode.step(parse_action(extract_json(reply), w["menu"], w["taxonomy"]))
    except (ValueError, KeyError, TypeError) as exc:  # DecodingError, EpisodeError, pydantic
        episode.state.done = True
        episode.state.termination_reason = "decode_failure"
        return f"{type(exc).__name__}: {exc}"[:500]
    return None


class DxData(vf.TaskData):
    split: str
    patient_id: str
    episode_seed: int
    budget: float
    max_turns: int


class DxTask(vf.Task[DxData]):
    """The seed task. Scoring happens in `DxEnv.run`, which plays the episode."""


class DxTasksetConfig(vf.TasksetConfig):
    split: Split = "eval"
    """train (14,854 patients), eval (3,713, frozen by hash), holdout (1,433 patients from
    two organ systems absent from train/eval)."""
    num_tasks: int | None = None
    """First n patients of the split (defaults to all)."""
    budget: float | None = None
    """Fixed test budget. Default: drawn per patient from the source p(B) over
    {10, 25, 50, 100, 200} with weights {.15, .25, .30, .20, .10}."""
    max_turns: int = 20
    """Turn cap. 20 is the source default; its GRPO curriculum started at 12."""


class DxTaskset(vf.Taskset[DxTask, DxTasksetConfig]):
    def load(self):
        c = self.config
        ids = _corpus()[1][c.split]
        n = len(ids) if c.num_tasks is None else min(c.num_tasks, len(ids))
        cfg = _world()["episode_config"]
        for i, pid in enumerate(ids[:n]):
            budget = c.budget
            if budget is None:
                budget = sample_budget(
                    cfg, np.random.default_rng([BUDGET_SEED, _SPLIT_CODE[c.split], i])
                )
            yield DxTask(
                DxData(
                    id=f"{c.split}-{i}",
                    name=f"{c.split}#{i} ({pid}, budget {budget:g})",
                    split=c.split,
                    patient_id=pid,
                    episode_seed=i,
                    budget=float(budget),
                    max_turns=c.max_turns,
                ),
                c.task,
            )


class DxEnvConfig(vf.EnvConfig):
    diagnostician: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "null"},
        runtime={"type": "subprocess"},
        sampling={"response_format": response_format()},
    )
    """One tool-less model call per turn, constrained to the source action schema. The null
    harness executes no code, so the local subprocess runtime is safe and avoids
    provisioning a sandbox for every turn."""


class DxEnv(vf.Env[DxEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        d = task.data
        w = _world()
        record = _corpus()[0][d.patient_id]
        episode = new_episode(record, d.episode_seed, d.budget, d.max_turns)
        obs = episode.reset()
        traces: list[vf.Trace] = []
        decode_error = None
        while not episode.state.done:
            turn = vf.Task(
                vf.TaskData(
                    idx=d.idx,
                    id=f"{d.id}/turn{episode.state.turn + 1}",
                    system_prompt=system_prompt(),
                    prompt=build_prompt(
                        obs, menu=w["menu"], taxonomy=w["taxonomy"], catalog=w["catalog"]
                    ),
                )
            )
            trace = await agents.diagnostician.run(turn)
            if not trace.ok:
                # A provider or runtime failure is not a policy failure: fail the episode
                # (retryable) rather than scoring it as a decode failure.
                raise RuntimeError(
                    f"diagnostician turn {episode.state.turn + 1} failed: {trace.errors[-1:]}"
                )
            traces.append(trace)
            decode_error = apply_reply(episode, trace.last_reply or "")
            obs = episode.observe()

        reward, facts = score(episode.trajectory(), record)
        reason = facts["termination_reason"]
        facts |= {
            "id": d.id,
            "patient_id": d.patient_id,
            "true_condition": record.condition,
            "budget": d.budget,
            "spent": episode.state.spent,
            "decode_error": decode_error,
        }
        for i, trace in enumerate(traces):
            final = i == len(traces) - 1
            trace.record_reward("diagnostic", reward)
            trace.record_metrics(
                {
                    "source_reward": facts["source_reward"],
                    "diagnosis": facts["diagnosis"],
                    "test_cost": facts["test_cost"],
                    "turn_penalty": facts["turn_penalty"],
                    "treatment": facts["treatment_hub"],
                    "coherence_adjustment": facts["coherence_adjustment"],
                    "verify": facts["verify"],
                    "n_tests": float(facts["n_tests_charged"]),
                    "mass_on_truth": facts["mass_on_truth"],
                    "top1_correct": float(facts["top1_correct"]),
                    "diagnosed": float(reason == "diagnose"),
                    "abstained": float(reason == "abstain"),
                    "out_of_turns": float(reason == "max_turns"),
                    "decode_failure": float(reason == "decode_failure"),
                    "n_contraindications": float(len(facts["contraindications"])),
                    "final_step": float(final),
                }
            )
            trace.info["dxenv"] = {**facts, "turn": i + 1, "final_step": final}


__all__ = ["DxEnv", "DxTaskset"]
