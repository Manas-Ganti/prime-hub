# dxenv: synthetic diagnostician (source: `healthcare_RL/`)

Repo: `git@github.com:Manas-Ganti/healthcare-RL.git` (public). Last commit `eb479ef` (2026-09-10).
Proposed Hub name: **`dxenv-diagnostician`** (not `synthea-diagnostician`; see Data).

## 1. Task

One episode is one synthetic patient. The agent sees an allowlisted, scrubbed observation:
demographics, free vitals, allergy list, remaining budget, turns remaining, results revealed so
far. It sees the same global action menu every turn (I3; ~100 tests/panels plus treatments).

Actions (one per turn, JSON; `dxenv/env/schemas.py`, executed in `dxenv/env/episode.py:167-195`):
- `order_test{test_key, prediction}`: costs budget. A coarse prediction of the headline result
  (low/normal/high or normal/abnormal) is **mandatory** and is committed before the reveal.
  Repeat orders are free and return the cached result (`episode.py:221-231`). Unaffordable
  orders are refused but still use the turn (`episode.py:233-240`).
- `prescribe{treatment_key}`
- `diagnose{distribution}`: terminal. A probability distribution over 149 flat labels.
- `abstain`: terminal.

Termination: diagnose, abstain, or `max_turns` (20 by default; the curriculum uses 12).
Budget is drawn per episode from {10, 25, 50, 100, 200} (`configs/env.yaml`).

## 2. Reward

Pure function `score_trajectory(trajectory, ground_truth, config)`,
`dxenv/reward/engine.py:216-339`. Total at `engine.py:319`:

```
R = diagnosis + test_cost + turn_penalty + treatment + shaping + verify
```

| term | formula | source | default weights |
|---|---|---|---|
| diagnosis | `[(1-1/n) - Σ(p_i - 1[i=c])²] · w_sev(c_true)`; shifted Brier, uniform report = 0. Range for n=149: about [-1.007, +0.993] × w | `reward/scoring.py:73-88, 123-132` | w ∈ {1.0, 1.8, 3.2, 6.0} by urgency tier of the **true** condition (`configs/severity.yaml`) |
| abstain / timeout | `abstain_value - abstain_penalty` = 0.0 - 0.05 | `engine.py:286-295` | |
| test_cost | `-λ · price(t)`, charged orders only | `reward/costs.py:61-72` | λ = 0.004 |
| turn_penalty | `-μ · n_turns` | `costs.py:75-81` | μ = 0.02 |
| treatment | `0.5·coherence + 1.0·correctness - 4.0·Σ contraindication scale` | `reward/treatment.py:141-177` | |
| verify (predict-then-verify) | `0.25 · λ·price · (1[correct bucket] - P_prior(bucket))`. Zero in expectation at chance; always below the order's own cost | `reward/verify.py:116-150` | fraction 0.25 |
| shaping | potential-based, Φ = -H(posterior), γ = 1. **Disabled by default** | `reward/shaping.py`, `engine.py:312-317` | scale 0.0008 |

- Terminal vs shaped: every term is computed once at episode end from the stored trajectory.
  The environment returns no per-step reward (`episode.py:3-10`).
- Shaping is potential-based (Ng et al. 1999), so it preserves the optimal policy. It is off
  because at a meaningful scale it would violate I5 ("tests never pay"), and
  `validate_reward_config` refuses such configs (`engine.py:137-172`).
- Logged, not trained on: the breakdown fields (`RewardBreakdown`, `engine.py:175-204`), the
  Bayes ceilings (`env/bayes.py`), calibration and Pareto metrics (`dxenv/eval/`).

## 3. Verifiability

Fully programmatic against hidden ground truth: no judge and no learned model. Trust comes from:
- 12 invariants with dedicated tests. Examples: the observation type has no label field (I1);
  the global menu (I3); every test returns a value for every patient (I4); tests never yield
  net-positive reward (I5); a strictly proper scoring rule (I7).
- A computable **Bayes-optimal ceiling** (`env/bayes.py`). Mean expected ceiling 1.722
  (`runs/phase3/prompted_baseline.json`). An episode above the hard ceiling indicates a leak.
- Caveat: the likelihoods are invented, and the Bayes baselines use the same generative model
  as the data. The env measures learning of a synthetic mapping, not clinical skill. The source
  README says this, and the Hub README must repeat it.

## 4. Reward-hacking audit

**Exploit found: the treatment coherence term pays without information (HIGH).**
`coherence` is the appropriateness of the prescriptions for the agent's *own declared argmax*
(`treatment.py:161-163`). It does not depend on the truth and is weighted 0.5. A policy that
orders nothing, reports the population prior, and prescribes one first-line drug for the prior's
argmax (essential hypertension → amlodipine) scores:

| policy (n=2000, source reward code) | mean R |
|---|---|
| prior, no prescription (the "blank-record floor") | -0.019 |
| **prior + 1 coherent prescription** | **+0.462** |
| reference: prompted Qwen2.5-7B / SFT / GRPO | -0.689 / -0.664 / -0.569 |
| reference: vitals-only Bayes | +0.675 |

Test: `reports/exploit_probes/test_dxenv_coherence_exploit.py` (passes, i.e. the exploit is live).
For context: no trained arm in the ARC work clears the blank-record floor on mean reward, and
this free +0.48 is larger than the gap between the floor and the vitals-only Bayes baseline.
Without a fix, a Hub user training on this env will probably find it within a few steps.

**Decision (2026-10-18): option 1, implemented in `environments/dxenv-diagnostician`.** Measured on 2000 patients
(source reward vs fixed): prior + coherent rx +0.462 → −0.030 (floor −0.019); prior shifted +0.3 to the
prescribed condition +0.283 → −0.204; shotgun top-5 first-line −0.110 → −0.209. Parse failures map to the
source's own `decode_failure` termination (scored −0.05, like a timeout).

Fix options considered (any of them breaks reward parity with the ARC runs):
1. Gate coherence on correctness: `coherence × mass_on_truth`.
2. Make coherence penalty-only (incoherent prescriptions lose, coherent ones earn 0).
3. Drop `coherence_scale` to ~0.05 so it is a tiebreaker.
4. Ship as-is with the exploit documented (not recommended for a public env).

Other probes:
- Constant / degenerate report: Brier is strictly proper, and uniform scores exactly 0. Abstain
  is priced at -0.05. Not exploitable.
- Name the high-severity condition: the weight is keyed on the **true** condition
  (`scoring.py:110-120`). Not exploitable.
- Spam cheap tests: repeat orders are free but earn no verify credit, because nothing new is
  charged (`engine.py:251-270`). Every charged test is net negative (I5, enforced arithmetically).
- Verify farming: zero expected value at chance and bounded below the order's cost. Not exploitable.
- Contraindications are scored against **ground-truth** analytes (`treatment.py:119-120`), not
  observed ones. A penalty can fire for a lab the agent never ordered, which contradicts the
  module docstring ("checkable by the agent"). This is noise, not an exploit. Note it in the README.
- Label leakage via prompt: covered by I1/I2 tests (`test_prompt_contains_no_label_string`).
  Not re-audited here.
- Format: on ARC, constrained decoding made invalid output impossible. On the Hub, API models
  emit free JSON. The port needs an explicit parse-failure rule (map it to the timeout/abstain
  value, or to a format penalty). This is a new design choice and must be documented.

## 5. Data

Fully procedural: `dxenv.data.corpus.generate_corpus(n, seed)`, the project's own disease-module
generator. **No Synthea data is used**: `data/snomed_map.yaml` is empty, so real Synthea output
cannot be ingested. The taxonomy (149 labels), catalog, costs and treatment tables are small
committed YAML files and can be vendored. The frozen eval split is defined by hash
(`data/eval_split.json`: corpus seed 20260901, 20k patients, 14854 train / 3713 eval / 1433
holdout on the rheumatologic + obstetric systems). It regenerates in `load()`. No license
problems. Calling it "Synthea" on the Hub would be inaccurate.

## 6. Evals and ARC results (separate harness)

| measurement | setup | headline | file |
|---|---|---|---|
| Gate A feasibility | 10k patients, seed 7; linear probes on features | top-1: vitals 0.463, tests 0.786, leak control 0.985 | `runs/phase0/results.json` |
| Gate B prompted baseline | Qwen2.5-7B-Instruct, n=200, k=8, T=1.0, seed 20260903, 20 turns | mean R -0.689, best@8 -0.480, schema-valid 1.0. Reference policies: prior -0.018, vitals-Bayes +0.675, greedy-Bayes +1.183 (5.76 tests) | `runs/phase3/prompted_baseline.json` |
| SFT / GRPO Gate B | same instrument | SFT -0.664; GRPO -0.569, best@8 +0.034, pass@8 0.330 → 0.105 | `docs/design-notes.md:409-421` (the raw SFT/GRPO JSONs are **not** on disk; only the doc table) |
| GRPO training | 99 steps, Qwen2.5-7B + LoRA from SFT, 8 patients × 8 samples, curriculum stage `single_condition_short` | tests/episode 0.79 → 3.5; reward plateau after step ~20 | `runs/grpo/steps.jsonl`, `runs/grpo/curve.png` |

## 7. Hub fit verdict: **Native**

Multi-turn LLM agent, programmatic reward, procedural data, a computable ceiling. It is the best
Hub candidate in the portfolio on paper. **Blocker before publishing: the coherence exploit.**
Port shape: Taskset + Toolset (`order_test`, `prescribe`, `diagnose`, `abstain` as tools) or a
JSON-action harness. Knobs: split (train/eval/holdout), num_tasks, budget (fixed or sampled),
max_turns, stage. The reward-parity test can call `score_trajectory` on stored trajectories directly.
