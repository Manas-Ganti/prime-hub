# Duration-aware bond allocation (source: `rl-bondportfolio/`)

Repo: `git@github.com:Manas-Ganti/bond-duration-allocation.git` (public). Last commit `d54ab49`.

## 1. Task (original)
`DurationAllocEnv` (`src/duralloc/env.py:114-170`). Monthly rebalancing over 24-month episodes
across maturity buckets under a two-factor (level, slope) curve model. Discrete action = an index
into a set of compliant allocations, with an action mask for the turnover / band / position limits.

## 2. Reward (original)
Per step: `carry + duration P&L - cost - penalty` (`env.py:92-111`). Cost is in bp per unit
traded. A penalty is applied to limit violations. The exact DP optimum is computed on a grid
(`problem.py`, `grid.py`).

## 3. Verifiability
Excellent. The DP optimum is exact in-model. The learned FQI policy matches it (-0.09 bp/yr,
98.4% within one grid step). Gates are in `GATES.md` / `results/gates.json`.

## 4. Reward-hacking audit (for an LLM wrapper)
- Per-step LLM decisions over 24 months: the reward is noisy per path, as with optimal stopping.
- Policy-synthesis framing: output an allocation rule over the (level, slope, current weights)
  state, scored by MC value vs DP on common paths. Verifiable, but the action/state encoding is
  more awkward than for optimal stopping, and the "right answer" is a 3-D lookup table. That is a
  harder sell as an LLM task.
- An always-static policy is a known baseline 49.9 bp/yr below DP, so degenerate answers score low.

## 5. Data
Simulation for training. FRED curves (`data/fred.py`) are used only for evaluation and
calibration. FRED data is mostly public domain, but keep the Hub env simulation-only.

## 6. Results (separate harness; `results/metrics.json`, seed 20261014)
DP 508.1 vs learned 508.1 vs static 458.2 bp/yr (5 bp cost). Cost sweep: the dynamic advantage
survives to 500 bp; myopic break-even at 75 bp.

## 7. Hub fit verdict: **Wrappable, weak**
The same idea as optimal stopping with a clunkier interface. Package optimal stopping first. If
it works, a bond variant can reuse the "synthesise a policy, score against DP" pattern. Low priority.
