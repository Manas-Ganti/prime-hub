"""The Hub env must measure what the original environment measured, except for the one
documented change (the coherence fix).

Replays 62 stored source episodes (healthcare_RL runs/phase3_prompted: prior, vitals-only
Bayes, greedy Bayes and the format-valid random policy, including every random episode
with a prescription) through the real `DxEnv.run` as model replies, and checks:

1. the trajectory is the source trajectory: same actions, costs, refusals, reveals,
   termination;
2. the logged `source_reward` equals the reward the source stored for that episode;
3. the Hub reward equals source reward + the coherence adjustment, and the adjustment is
   exactly zero for every episode without both a prescription and a diagnosis.
"""

import json
from pathlib import Path

import pytest
from dxenv.data.splits import load_frozen_provenance
from dxenv.reward.engine import load_reward_config

FIXTURES = Path(__file__).parent / "fixtures" / "source_episodes.jsonl"
ROWS = [json.loads(line) for line in FIXTURES.read_text().splitlines()]


@pytest.mark.parametrize("row", ROWS, ids=[f"{r['policy']}:{r['patient_id'][-3:]}:{i}" for i, r in enumerate(ROWS)])
def test_replayed_episode_reproduces_source(row, play):
    traces = play(row["patient_id"], row["replies"], budget=row["budget"], seed=row["seed"])
    facts = traces[-1].info["dxenv"]

    assert len(traces) == len(row["steps"])
    assert facts["termination_reason"] == row["termination_reason"]
    assert facts["true_condition"] == row["condition"]
    assert facts["n_turns"] == len(row["steps"])
    assert facts["source_reward"] == pytest.approx(row["source_reward"], abs=1e-9)

    prescribed = any(s["kind"] == "prescribe" for s in row["steps"])
    if not (prescribed and row["termination_reason"] == "diagnose"):
        assert facts["coherence_adjustment"] == 0.0
    expected = row["source_reward"] + facts["coherence_adjustment"]
    for t in traces:
        assert t.rewards["diagnostic"] == pytest.approx(expected, abs=1e-9)
    assert [t.metrics["final_step"] for t in traces] == [0.0] * (len(traces) - 1) + [1.0]


def test_fixture_exercises_the_coherence_fix_and_refusals():
    with_fix = [r for r in ROWS if r["termination_reason"] == "diagnose" and any(s["kind"] == "prescribe" for s in r["steps"])]
    assert len(with_fix) >= 10
    assert {r["termination_reason"] for r in ROWS} == {"diagnose", "abstain"}
    assert any(s["kind"] == "order_test" and s["cost_charged"] == 0.0 for r in ROWS for s in r["steps"])


def test_coherence_adjustment_matches_its_formula(play, fixture_records):
    """Adjustment == -0.5 * coherence * (1 - mass_on_truth), from first principles."""
    from dxenv.reward.treatment import load_treatment_config

    tc = load_treatment_config()
    pid, rec = next(iter(fixture_records.items()))
    wrong = next(c for c in tc.first_line if c != rec.condition)
    rx = tc.first_line[wrong][0]
    dist = [{"condition": wrong, "probability": 0.7}, {"condition": rec.condition, "probability": 0.2}]
    replies = [
        json.dumps({"kind": "prescribe", "reasoning": "", "treatment_key": rx}),
        json.dumps({"kind": "diagnose", "reasoning": "", "diagnosis": dist}),
    ]
    facts = play(pid, replies)[-1].info["dxenv"]
    coherence = tc.appropriateness(rx, wrong)
    assert coherence > 0
    assert facts["mass_on_truth"] == pytest.approx(0.2)
    assert facts["coherence_adjustment"] == pytest.approx(-0.5 * coherence * 0.8)


def test_reward_config_is_the_source_default():
    cfg = load_reward_config()
    assert (cfg.lam, cfg.mu, cfg.coherence_scale, cfg.correctness_scale) == (0.004, 0.02, 0.5, 1.0)
    assert not cfg.shaping_enabled and cfg.verify_enabled
    assert load_frozen_provenance()["corpus_seed"] == 20260901
