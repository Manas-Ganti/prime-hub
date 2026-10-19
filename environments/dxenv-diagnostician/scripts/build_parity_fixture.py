"""Build tests/fixtures/source_episodes.jsonl from the source repo's stored episodes.

Source: healthcare_RL/runs/phase3_prompted/episodes.jsonl (Gate B reference sweep: 40
patients from generate_corpus(40, seed=20260903), policies prior / vitals_bayes /
greedy_bayes / random_schema). Each stored action is rendered back into the wire JSON a
model would emit, so the parity test can replay it through the Hub env as model replies.
Run from this package directory with the source repo next to the workspace:

    python scripts/build_parity_fixture.py
"""

import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[4] / "healthcare_RL" / "runs" / "phase3_prompted" / "episodes.jsonl"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "source_episodes.jsonl"
PER_POLICY = {"prior": 4, "vitals_bayes": 4, "greedy_bayes": 6}


def wire(action: dict) -> str:
    kind = action["kind"]
    obj = {"kind": kind, "reasoning": "replayed from a stored source trajectory"}
    if kind == "order_test":
        obj |= {"test_key": action["test_key"], "prediction": action["prediction"]}
    elif kind == "prescribe":
        obj |= {"treatment_key": action["treatment_key"]}
    elif kind == "diagnose":
        obj |= {"diagnosis": [{"condition": c, "probability": p} for c, p in action["distribution"].items()]}
    return json.dumps(obj, separators=(",", ":"))


def main() -> None:
    eps = [json.loads(line) for line in SRC.read_text().splitlines()]
    picked, counts = [], dict.fromkeys(PER_POLICY, 0)
    for e in eps:
        pol = e["tags"]["policy"]
        steps = e["trajectory"]["steps"]
        if pol == "random_schema":
            # every episode with a prescription (exercises the coherence fix), plus refusals
            keep = any(s["action"]["kind"] == "prescribe" for s in steps)
        else:
            keep = counts[pol] < PER_POLICY[pol]
            counts[pol] += keep
        if keep:
            picked.append(
                {
                    "policy": pol,
                    "patient_id": e["ground_truth"]["patient_id"],
                    "condition": e["ground_truth"]["condition"],
                    "seed": e["trajectory"]["seed"],
                    "budget": e["trajectory"]["budget"],
                    "termination_reason": e["trajectory"]["termination_reason"],
                    "source_reward": e["tags"]["reward"],
                    "replies": [wire(s["action"]) for s in steps],
                    "steps": [
                        {
                            "kind": s["action"]["kind"],
                            "key": s["action"].get("test_key") or s["action"].get("treatment_key"),
                            "cost_charged": s.get("cost_charged", 0.0),
                            "was_duplicate": s.get("was_duplicate", False),
                            "n_revealed": len(s.get("revealed", [])),
                        }
                        for s in steps
                    ],
                }
            )
    OUT.write_text("".join(json.dumps(p) + "\n" for p in picked))
    print(f"wrote {len(picked)} episodes to {OUT}")


if __name__ == "__main__":
    main()
