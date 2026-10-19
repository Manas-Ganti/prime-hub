# Optimal stopping for American options (source: `finance-rl/`)

Repo: `git@github.com:Manas-Ganti/optimal-stopping-rl.git` (public). Last commit `60e67f1`.

## 1. Task (original)
Gym env `OptionStoppingEnv` (`src/optstop/env.py:30-80`). Each step at one of 50 exercise dates
the agent sees (normalised spot, time) and chooses exercise or hold. The policy is fitted Q-iteration
with RBF features (`agent/fitted_q.py`), checked against Longstaff-Schwartz and an exact lattice.

## 2. Reward (original)
On exercise: `exp(-r t_k) · max(K - S, 0) / K`; otherwise 0. At expiry, forced exercise
(`env.py:5-7, 60-63`). The undiscounted episode return equals the discounted payoff, so the
optimal policy maximises option value. A per-path reward is very noisy; value is estimated on
200k common paths.

## 3. Verifiability
Excellent. The optimum is known (exact Bermudan price from the lattice, LSM as a reference), and
Gates A–E are pre-registered (`GATES.md`, `results/gates.json`).

## 4. Reward-hacking audit (for an LLM wrapper)
- Per-step LLM version (50 turns of "exercise/hold"): the single-path reward is dominated by
  path noise. Group-relative advantages would mostly measure luck. Not a good training signal.
- Better framing: **single-turn policy synthesis**. The agent outputs an exercise boundary
  (50 numbers, or a small parametric rule). Reward = Monte-Carlo value of that boundary on N fixed
  common paths ÷ exact price (or regret vs LSM). Low noise, computable optimum, and it can't be
  gamed. Main risk: degenerate boundaries (always exercise / never exercise) score low by
  construction (hold-to-expiry is 0.19–0.31 below the optimum).

## 5. Data
Simulation only (GBM / Merton / Heston, seeded). The real-data test uses market prices through
`data/fetch.py` (yfinance-style). Those prices can't be redistributed, so the Hub env should be
simulation-only.

## 6. Results (separate harness; `results/metrics.json`)
Learned 2.3107 ± 0.0036 vs LSM 2.3104 ± 0.0040 vs exact 2.3141 (K=40, S=40, σ=0.2, r=0.06, T=1).
Boundary within 0.015 S/K on average of LSM. Misspecification regret (Merton/Heston) and real-data
windows (384) are in the README. None of this transfers to an LLM benchmark.

## 7. Hub fit verdict: **Wrappable** (creates a new benchmark)
As "derive the exercise boundary for this option" with exact-price scoring, it is a genuinely
interesting quantitative-reasoning env with a verifiable optimum and procedural parameters (K, σ, r,
T, dates, dynamics). It is a **new** benchmark, and the FQI results don't apply. Medium priority.
It's distinctive on the Hub, but it's a design project, not a port.
