"""The Hub env must measure exactly what the original environment measured.

1. Instances regenerated from (split, index) are the instances in the source repo's data/v1
   (checked against the local file when the source repo sits next to this workspace).
2. Replaying the source repo's stored trajectories (results/eval_*.trajectories.jsonl: the
   privileged oracle, the random reader and the no-read policy, ID and OOD) through the Hub
   `NavigatorEnv.run` reproduces each episode's original reward and outcome flags exactly.
"""

import json
from pathlib import Path

import pytest

from long_context_navigator.taskset import build_instance, split_params

FIXTURES = Path(__file__).parent / "fixtures" / "source_trajectories.jsonl"
SOURCE_DATA = Path(__file__).resolve().parents[4] / "long-context-navigator-RL" / "data" / "v1"
ADDED_SINCE_V1 = {"substrate": "synthetic", "answer_aliases": []}  # schema defaults added later

ROWS = [json.loads(line) for line in FIXTURES.read_text().splitlines()]


def _index(instance_id: str, split: str) -> int:
    from long_context_navigator.taskset import _configs

    seed = int(instance_id.rsplit("-", 1)[1]) // 1000
    return seed - _configs()[1].splits[split].seed_base


@pytest.mark.parametrize("split", ["id_test", "ood_test", "scale_test"])
def test_regenerated_instances_match_source_data(split):
    path = SOURCE_DATA / f"{split}.jsonl"
    if not path.exists():
        pytest.skip("source repo data/v1 not available next to this workspace")
    with path.open() as f:
        for i in range(3):
            ref = json.loads(f.readline())
            got = build_instance(split, *split_params(split, i)).model_dump(mode="json")
            for k, v in ADDED_SINCE_V1.items():
                assert got.pop(k) == v
            assert got == ref


@pytest.mark.parametrize("row", ROWS, ids=[f"{r['policy']}:{r['instance_id']}" for r in ROWS])
def test_replayed_trajectory_reproduces_original_reward(row, play_episode):
    traces = play_episode(row["split"], _index(row["instance_id"], row["split"]), row["actions"])

    assert len(traces) == row["steps_used"]
    facts = traces[-1].info["navigator"]
    for trace in traces:
        assert trace.rewards["navigation"] == pytest.approx(row["reward"], abs=1e-12)
    assert facts["correct"] == row["correct"]
    assert facts["answered"] == row["answered"]
    assert facts["ceiling_exceeded"] == row["ceiling_exceeded"]
    assert facts["budget_exhausted"] == row["budget_exhausted"]
    assert facts["n_compress"] == row["n_compress"]
    assert facts["ground_truth"] == build_instance(
        row["split"], *split_params(row["split"], _index(row["instance_id"], row["split"]))
    ).answer
    assert [t.metrics["final_step"] for t in traces] == [0.0] * (len(traces) - 1) + [1.0]


def test_fixture_covers_correct_and_wrong_and_both_splits():
    assert {r["split"] for r in ROWS} == {"id_test", "ood_test"}
    assert {r["correct"] for r in ROWS} == {True, False}
    assert {r["policy"] for r in ROWS} == {"oracle", "random-reader", "no-read"}
