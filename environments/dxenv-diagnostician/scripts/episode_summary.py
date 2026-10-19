"""Per-EPISODE summary of a vf-eval run directory.

Each diagnostic turn is its own trace, so vf-eval's headline (a mean over traces) weights
long episodes more heavily. This reads <run_dir>/traces.jsonl and reports one row per
episode, using the final turn's facts.

    python scripts/episode_summary.py outputs/dxenv-diagnostician--<model>--null--<id>
"""

import json
import math
import sys
from pathlib import Path


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n)


def main(run_dir: str) -> None:
    episodes, errored = [], 0
    for line in (Path(run_dir) / "traces.jsonl").read_text().splitlines():
        ep = json.loads(line)
        if not ep.get("ok") or not ep.get("traces"):
            errored += 1
            continue
        f = ep["traces"][-1]["info"]["dxenv"]
        r = f["termination_reason"]
        episodes.append(
            {
                "reward (hub)": f["total"] + f["coherence_adjustment"],
                "source_reward": f["source_reward"],
                "diagnosis": f["diagnosis"],
                "mass_on_truth": f["mass_on_truth"],
                "top1_correct": float(f["top1_correct"]),
                "n_tests": float(f["n_tests_charged"]),
                "n_turns": float(f["n_turns"]),
                "diagnosed": float(r == "diagnose"),
                "abstained": float(r == "abstain"),
                "out_of_turns": float(r == "max_turns"),
                "decode_failure": float(r == "decode_failure"),
            }
        )
    if not episodes:
        sys.exit(f"no completed episodes ({errored} errored)")
    print(f"episodes: {len(episodes)} completed, {errored} errored")
    for key in episodes[0]:
        m, se = mean_se([e[key] for e in episodes])
        print(f"  {key:15s} {m:7.3f} ± {se:.3f} (SE)")


if __name__ == "__main__":
    main(sys.argv[1])
