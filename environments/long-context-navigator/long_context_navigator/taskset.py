"""long-context-navigator: multi-hop QA over a document far larger than the working context.

The agent answers a k-hop question over a ~32k-token synthetic document while holding at most
600 tokens of it. Chunks are 320-480 tokens, so two never fit together: every bridge entity
has to be carried across reads in a COMPRESS summary. Any action that would push the working
context over the ceiling ends the episode as a failure.

The policy is STATELESS between steps, as in the original environment: every step is a fresh
single-turn rollout whose prompt is the rendered environment state (question, document map,
memory ledger, working context). Chat history is never carried forward, otherwise the model
would "remember" chunks it dropped and the ceiling would mean nothing. Each step is one trace;
every trace of an episode carries the episode's return, so a consumer can inherit one
advantage across all steps exactly as the original GRPO loop did.

Environment mechanics and the reward are imported unchanged from the source repo
(`longctx.env.NavigationEnv`, `longctx.reward.compute_reward`), so the Hub reward is the
original reward by construction; `tests/test_reward_parity.py` checks it anyway.
"""

import random
from functools import cache
from importlib.resources import files
from typing import Literal

import yaml
from longctx.config import EnvConfig, GeneratorConfig
from longctx.env import NavigationEnv
from longctx.generate import InstanceGenerator
from longctx.prompts import SYSTEM_PROMPT, render_observation
from longctx.reward import answer_matches
from longctx.schema import Instance

import verifiers.v1 as vf

Split = Literal["train", "id_test", "ood_test", "scale_test"]


@cache
def _configs() -> tuple[EnvConfig, GeneratorConfig]:
    """The source repo's env.yaml / generator.yaml, vendored as package data (the source
    package reads them from its repo root, which a wheel install does not have)."""
    root = files("long_context_navigator").joinpath("configs")
    env = EnvConfig(**yaml.safe_load(root.joinpath("env.yaml").read_text()))
    gen = GeneratorConfig(**yaml.safe_load(root.joinpath("generator.yaml").read_text()))
    return env, gen


@cache
def _generator() -> InstanceGenerator:
    env, gen = _configs()
    return InstanceGenerator(gen, env)


def build_instance(split: str, seed: int, n_hops: int, doc_tokens: int) -> Instance:
    """Deterministic: the same (split, seed, n_hops, doc_tokens) gives the same instance
    the source repo's `longctx generate` writes to data/v1."""
    inst, _ = _generator().generate_valid(seed, n_hops, doc_tokens, split)
    return inst


def split_params(split: str, i: int) -> tuple[int, int, int]:
    """(seed, n_hops, doc_tokens) for the i-th instance of a split, mirroring
    `longctx.generate.generate_split`."""
    cfg = _configs()[1].splits[split]
    seed = cfg.seed_base + i
    rng = random.Random(seed)
    n_hops = rng.choice(cfg.n_hops)
    doc_tokens = rng.choice(cfg.doc_tokens)
    return seed, n_hops, doc_tokens


class NavigatorData(vf.TaskData):
    split: str
    seed: int
    n_hops: int
    doc_tokens: int


class NavigatorTask(vf.Task[NavigatorData]):
    """The seed task. Scoring happens in `NavigatorEnv.run`, which plays the episode."""


class NavigatorTasksetConfig(vf.TasksetConfig):
    split: Split = "id_test"
    """train (2-3 hops), id_test (2-3), ood_test (4-5, deeper than anything trained on),
    scale_test (3 hops at 32k / 64k / 128k tokens)."""
    num_tasks: int | None = None
    """Instances to generate; defaults to the split's size in generator.yaml (2000 / 300 /
    300 / 150). Larger values extend the split with fresh seeds."""


class NavigatorTaskset(vf.Taskset[NavigatorTask, NavigatorTasksetConfig]):
    def load(self):
        split = self.config.split
        n = self.config.num_tasks or _configs()[1].splits[split].n
        for i in range(n):
            seed, n_hops, doc_tokens = split_params(split, i)
            yield NavigatorTask(
                NavigatorData(
                    id=f"{split}-{i}",
                    name=f"{split}#{i} ({n_hops}-hop, {doc_tokens // 1000}k tokens)",
                    split=split,
                    seed=seed,
                    n_hops=n_hops,
                    doc_tokens=doc_tokens,
                ),
                self.config.task,
            )


class NavigatorEnvConfig(vf.EnvConfig):
    navigator: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "null"}, runtime={"type": "subprocess"}
    )
    """One tool-less model call per step. The null harness executes no code, so the local
    subprocess runtime is safe and avoids provisioning a sandbox for every step."""


class NavigatorEnv(vf.Env[NavigatorEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        d = task.data
        inst = build_instance(d.split, d.seed, d.n_hops, d.doc_tokens)
        env = NavigationEnv(_configs()[0])
        obs = env.reset(inst)
        traces: list[vf.Trace] = []
        while not env.done:
            step = vf.Task(
                vf.TaskData(
                    idx=d.idx,
                    id=f"{d.id}/step{env.steps_used + 1}",
                    system_prompt=SYSTEM_PROMPT,
                    prompt=render_observation(obs),
                )
            )
            trace = await agents.navigator.run(step)
            if not trace.ok:
                # A provider or runtime failure is not a navigation failure: fail the
                # episode (retryable) rather than scoring it as a wrong action.
                raise RuntimeError(
                    f"navigator step {env.steps_used + 1} failed: {trace.errors[-1:]}"
                )
            traces.append(trace)
            obs, _, _, _ = env.step(trace.last_reply)

        reward = env._final_reward()
        info = env.info()
        outcome = env.outcome()
        facts = {
            **info,  # its "answer" is the model's prediction; ground truth is stored apart
            "id": d.id,
            "n_hops": inst.n_hops,
            "doc_tokens": inst.doc_tokens,
            "question": inst.question,
            "ground_truth": inst.answer,
            "min_steps": inst.min_steps,
            "evidence_chunks": inst.evidence_chunks,
        }
        distractor_answers = {
            x.wrong_answer or x.value for x in inst.distractors if x.wrong_answer or x.value
        }
        wrong = outcome.answered and not info["correct"]
        for i, trace in enumerate(traces):
            final = i == len(traces) - 1
            trace.record_reward("navigation", reward)
            trace.record_metrics(
                {
                    "correct": float(info["correct"]),
                    "ceiling_exceeded": float(info["ceiling_exceeded"]),
                    "budget_exhausted": float(info["budget_exhausted"]),
                    "steps_over_min": info["steps_used"] / inst.min_steps,
                    "n_compress": float(info["n_compress"]),
                    "n_invalid": float(info["n_invalid"]),
                    "distractor_capture": float(
                        wrong
                        and any(
                            answer_matches(outcome.answer, a) for a in distractor_answers
                        )
                    ),
                    "final_step": float(final),
                }
            )
            trace.info["navigator"] = {**facts, "step": i + 1, "final_step": final}


__all__ = ["NavigatorEnv", "NavigatorTaskset"]
