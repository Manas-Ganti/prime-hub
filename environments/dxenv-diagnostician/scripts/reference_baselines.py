"""Model-free reference policies on the same patients and budgets a vf-eval run sees.

Scores the source repo's heuristic policies (dxenv.policy.baselines) through this
package's episode and reward code, on the first n tasks of a split:

  prior          report the population prior, order nothing (the blank-record floor)
  vitals_bayes   Bayes posterior on the free case sheet, order nothing
  greedy_bayes   order up to 6 tests by expected entropy per cost, then report the posterior

and the per-patient expected Bayes ceiling (every analyte revealed for free; an upper bound
on the diagnosis term). Usage:

    python scripts/reference_baselines.py --split eval -n 50 [--out outputs/reference_eval_50.json]
"""

import argparse
import json
import math
from pathlib import Path

from dxenv.policy.baselines import GreedyBayesPolicy, PriorPolicy, VitalsOnlyPolicy, run_episode
from dxenv.policy.rollout import RolloutContext

from dxenv_diagnostician.taskset import DxTaskset, DxTasksetConfig, _corpus, new_episode, score


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--out")
    args = ap.parse_args()

    tasks = [t.data for t in DxTaskset(DxTasksetConfig(split=args.split, num_tasks=args.n)).load()]
    records = _corpus()[0]
    policies = {"prior": PriorPolicy, "vitals_bayes": VitalsOnlyPolicy, "greedy_bayes": GreedyBayesPolicy}
    ctx = RolloutContext()

    rows = {name: [] for name in policies}
    ceilings = []
    for d in tasks:
        rec = records[d.patient_id]
        ceilings.append(ctx.expected_for(rec))
        for name, cls in policies.items():
            ep = new_episode(rec, d.episode_seed, d.budget, d.max_turns)
            reward, facts = score(run_episode(ep, cls()), rec)
            rows[name].append({"reward": reward, **{k: facts[k] for k in (
                "source_reward", "mass_on_truth", "top1_correct", "n_tests_charged")}})

    summary = {"split": args.split, "n": len(tasks), "policies": {}}
    print(f"{args.split}, first {len(tasks)} patients (same budgets as vf-eval):")
    for name, rs in rows.items():
        s = {k: mean_se([float(r[k]) for r in rs]) for k in rs[0]}
        summary["policies"][name] = s
        print(f"  {name:13s} reward {s['reward'][0]:+.3f} ± {s['reward'][1]:.3f}   top1 {s['top1_correct'][0]:.2f}"
              f"   mass_on_truth {s['mass_on_truth'][0]:.3f}   tests {s['n_tests_charged'][0]:.2f}")
    summary["expected_bayes_ceiling"] = mean_se(ceilings)
    print(f"  expected Bayes ceiling (diagnosis term, all evidence free): {summary['expected_bayes_ceiling'][0]:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
