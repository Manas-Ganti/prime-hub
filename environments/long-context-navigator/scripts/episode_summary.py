"""Per-EPISODE summary of a vf-eval run directory.

Each navigation step is its own trace, so vf-eval's headline (a mean over traces) weights long
episodes more heavily. This reads <run_dir>/traces.jsonl and reports one row per episode,
using the final step's facts.

    python scripts/episode_summary.py outputs/long-context-navigator--<model>--null--<id>
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n) if n else 0.0


def main(run_dir: str) -> None:
    episodes, errored = [], 0
    for line in (Path(run_dir) / "traces.jsonl").read_text().splitlines():
        ep = json.loads(line)
        if not ep.get("ok") or not ep.get("traces"):
            errored += 1
            continue
        last = ep["traces"][-1]
        facts = last["info"]["navigator"]
        episodes.append(
            {
                "reward": last["rewards"]["navigation"]["score"],
                "correct": float(facts["correct"]),
                "ceiling": float(facts["ceiling_exceeded"]),
                "budget": float(facts["budget_exhausted"]),
                "steps_over_min": facts["steps_used"] / facts["min_steps"],
                "compress": float(facts["n_compress"] > 0),
                "n_hops": facts["n_hops"],
            }
        )
    if not episodes:
        sys.exit(f"no completed episodes ({errored} errored)")

    print(f"episodes: {len(episodes)} completed, {errored} errored")
    for key in ("reward", "correct", "ceiling", "budget", "steps_over_min", "compress"):
        m, se = mean_se([e[key] for e in episodes])
        print(f"  {key:15s} {m:7.3f} ± {se:.3f} (SE)")
    by_hops = defaultdict(list)
    for e in episodes:
        by_hops[e["n_hops"]].append(e["correct"])
    print("  accuracy by n_hops: " + ", ".join(
        f"{h}: {sum(v) / len(v):.3f} (n={len(v)})" for h, v in sorted(by_hops.items())
    ))


if __name__ == "__main__":
    main(sys.argv[1])
