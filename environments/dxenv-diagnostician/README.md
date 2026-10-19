# dxenv-diagnostician

Cost-aware diagnosis over synthetic patients. The agent works one case at a time: it reads a
case sheet (demographics, complaint, vitals, history, allergies), orders tests that cost
budget, may prescribe, and ends the case by reporting a **probability distribution** over 149
conditions. It is scored by a strictly proper rule, so its best strategy is to report
honest, calibrated beliefs and to buy a test only when the information is worth the cost.

The patients are generated from a hand-written generative model, which is what makes a
**Bayes-optimal ceiling** computable for every patient. This environment measures how well an
agent learns a synthetic diagnostic mapping under cost. It does **not** measure clinical
skill. The disease likelihoods are invented, and no real patient data (Synthea or otherwise)
is used.

## Task and interaction

Each case uses a global menu, identical for every patient, of **73 tests/panels** and
**72 treatments**, plus the 149 condition labels. A per-patient menu would leak the diagnosis.
The agent takes one action per turn as a JSON object:

| action | effect |
|---|---|
| `order_test` + `prediction` | reveals the test's analytes and costs budget. The agent must first commit to a coarse prediction of the headline result (`low`/`normal`/`high`, or `normal_categorical`/`abnormal_categorical`) |
| `prescribe` | records a treatment |
| `diagnose` | terminal: up to 16 `{condition, probability}` pairs; unassigned mass is spread evenly over the unnamed labels |
| `abstain` | terminal |

Mechanics:
- Repeat orders are free and return the cached result.
- An order the remaining budget can't cover is refused, and the turn is still spent.
- The budget is drawn per patient from {10, 25, 50, 100, 200}, weighted .15/.25/.30/.20/.10.
- An episode ends on diagnose or abstain, or after 20 turns.

**The policy is stateless.** Each turn is a fresh single-turn model call. Its prompt is
rebuilt from the typed observation, which has no field that can carry the label. No chat
history is carried over, and each turn is recorded as its own trace.

**Output format (differs from the source).** On the original cluster, a JSON grammar made
invalid output impossible. Here the same action grammar is sent as a JSON-schema
`response_format` on the diagnostician's sampling args, and the wire format is also spelled
out in the system prompt. Providers that honour structured output (OpenAI-compatible APIs,
vLLM, Ollama) are then constrained the same way. Ones that ignore it fall back to the prompt.
A reply wrapped in a markdown fence or a `<think>` block is unwrapped, which changes no
field. A reply that still isn't one valid JSON object on the menu ends the case as
`decode_failure`. The source scores that as a non-decision (−0.05), the same as running out
of turns. Without the schema, Qwen2.5-7B failed to decode on 5 of 5 smoke episodes. With
it, 0 of 5 failed.

## Reward

One reward per episode, `diagnostic`, recorded on **every turn's trace**:

```
R = diagnosis + test_cost + turn_penalty + treatment + verify
```

| term | formula | notes |
|---|---|---|
| diagnosis | `w(c_true) · [(1 − 1/149) − Σ_i (p_i − 1[i = c_true])²]` | shifted Brier: a uniform report scores 0, a confident correct one about +w, a confident wrong one about −w. `w` ∈ {1.0, 1.8, 3.2, 6.0} by urgency of the **true** condition |
| abstain / out of turns / decode failure | diagnosis term = −0.05 | strictly worse than reporting the prior |
| test_cost | `−0.004 · price` for each charged order | repeat and refused orders cost nothing |
| turn_penalty | `−0.02 · n_turns` | |
| treatment | `0.5 · coherence · mass_on_truth + 1.0 · correctness − 4.0 · Σ contraindication scales` | see *Coherence fix* below |
| verify | `0.25 · order cost · (1[prediction right] − P_prior(bucket))` | zero in expectation at chance, always smaller than the order's own cost |

Properties, each checked by `tests/test_exploits.py`:
- Every test is net negative on its own, even with a correct prediction. Tests only pay
  through a better diagnosis.
- Constant predictions earn nothing.
- Naming severe conditions doesn't pay.

There is no judge. The reward is computed from hidden ground truth.

Metrics (logged, not rewarded):
- the separate terms;
- `source_reward`, the original reward with no coherence fix, for comparison with the
  original results;
- `mass_on_truth`, `top1_correct`, `n_tests`;
- the termination flags `diagnosed`, `abstained`, `out_of_turns`, `decode_failure`;
- `n_contraindications`, and `final_step` (1.0 on the episode's last trace).

### Coherence fix (this package differs from the original)

The original treatment term paid `0.5 · coherence`. Coherence is how well the prescriptions
fit the agent's **own** top diagnosis, whether or not that diagnosis is right. A policy that
orders nothing, reports the population prior, and always prescribes amlodipine (first-line
for the most common condition) averaged **+0.46** under the original reward. That beats every
trained model in the original work, with no information about the patient. Here the term
is gated on the probability the report puts on the truth:

```
original:  0.5 · coherence
this env:  0.5 · coherence · mass_on_truth
```

| policy (40 fixture patients, mean R) | original reward | this env |
|---|---|---|
| report the prior, order nothing (floor) | −0.019 | −0.019 |
| prior + one coherent prescription | **+0.464** | −0.028 |
| same, with 0.3 / 0.6 / 0.9 extra mass moved to the prescribed condition | +0.350 / −0.053 / −0.743 | −0.134 / −0.527 / −1.208 |

This is the only change to the reward arithmetic. It is applied as a delta on the original
score, and the original score is logged as `source_reward`.
`tests/test_reward_parity.py` replays 62 stored episodes from the source repo and checks
three things:
- the trajectories match;
- `source_reward` equals the stored reward;
- the Hub reward equals `source_reward` plus the adjustment, and the adjustment is exactly 0
  whenever an episode lacks a prescription or a diagnosis.

## Why the reward can be trusted

- **No label in the observation, by type.** The observation model has no field that can
  carry a condition. The display strings come from the catalog, never from the patient
  record. The test suite also checks that no case sheet contains its own label.
- **Fixed menu.** Every patient sees the same tests, treatments and labels.
- **Proper scoring.** The Brier rule is strictly proper, and the severity weight is keyed on
  the true condition.
- **Tests never pay on their own.** This is enforced arithmetically by the source repo's
  `validate_reward_config`.
- **A ceiling exists.** The generative model is known, so the source repo computes a
  per-patient Bayes-optimal ceiling. An episode scoring above it would indicate a leak.

## Data and license

The data is procedural, and nothing is downloaded or bundled. `load()` regenerates the
source repo's frozen 20,000-patient corpus (seed 20260901), which takes about 12 s once per
process. It then re-cuts the split and verifies it against the committed eval hash.

| split | patients | notes |
|---|---|---|
| `train` | 14,854 | |
| `eval` (default) | 3,713 | frozen by hash in the source repo |
| `holdout` | 1,433 | rheumatologic and obstetric conditions only, which never appear in train or eval |

The taxonomy, test catalog, costs and treatment tables are small YAML files inside the
`dxenv` package.

## Configuration

| flag | default | meaning |
|---|---|---|
| `--env.taskset.split` | `eval` | `train`, `eval` or `holdout` |
| `--env.taskset.num-tasks` | whole split | first n patients |
| `--env.taskset.budget` | sampled | fixed test budget instead of the per-patient draw |
| `--env.taskset.max-turns` | 20 | turn cap (the source GRPO curriculum started at 12) |
| `--env.diagnostician.model` / `.sampling.*` | the run's model | the diagnosing model |

Reward weights are fixed to the source values. Only the coherence term differs.

## Quickstart

```bash
prime env install manasganti/dxenv-diagnostician
uv run vf-eval dxenv-diagnostician -n 5 -r 1 -m <model>
uv run vf-eval dxenv-diagnostician -n 50 -r 1 -m <model>
python scripts/episode_summary.py outputs/dxenv-diagnostician--<model>--null--<id>
```

Each turn's prompt is about 3k tokens: the case sheet, the full menu and the label list.
Allow about 1k completion tokens per turn, since a diagnosis that names 16 conditions is
the longest reply.

**Reading eval output.** vf-eval's headline is a mean over *traces*, and an episode has one
trace per turn, so long episodes count more. `scripts/episode_summary.py` gives per-episode
numbers. For training, every turn's trace carries its episode's return, so group by episode
when computing advantages.

## Baseline results (vf-eval, this package)

*Pending: no vf-eval run against a real model has been recorded yet.*

| model | split | episodes | reward (mean ± SE) | source_reward | top-1 | tests / episode | decode failures |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

**Model-free reference policies** on the same first 50 patients and budgets a `-n 50` run
sees. These are the source repo's heuristic policies, scored by this package's reward
(`python scripts/reference_baselines.py --split <split> -n 50`). The Bayes policies use the
true generative model, so they are references, not competitors.

| policy | eval: reward (± SE) | eval: top-1 | holdout: reward (± SE) | holdout: top-1 | tests / episode |
|---|---|---|---|---|---|
| report the prior, order nothing (floor) | −0.019 ± 0.007 | 0.08 | −0.019 ± 0.006 | 0.00 | 0 |
| vitals-only Bayes, order nothing | +1.221 ± 0.242 | 0.60 | +0.521 ± 0.168 | 0.58 | 0 |
| greedy Bayes (up to 6 tests) | +1.764 ± 0.272 | 0.78 | +0.783 ± 0.199 | 0.76 | 5.7 |
| expected Bayes ceiling (diagnosis term, all evidence free) | 2.244 | | 1.397 | | |

## Original results (separate harness)

These come from the source repo's own harness on VT ARC (vLLM with grammar-constrained
decoding, Qwen2.5-7B-Instruct). They were **not** produced by this package or by vf-eval, and
they are **scored under the original reward, before the coherence fix**. They are not Hub
baselines.

Gate B protocol: 200 train-split patients, k = 8 samples, T = 1.0, 20 turns, budget drawn per
patient.

| policy | mean reward | source |
|---|---|---|
| report the prior (blank-record floor) | −0.018 | `runs/phase3/prompted_baseline.json` |
| vitals-only Bayes (no tests) | +0.675 | same |
| greedy Bayes (5.76 tests on average) | +1.183 | same |
| Qwen2.5-7B-Instruct, prompted | −0.689 (best@8 −0.480) | same |
| + SFT | −0.664 | `docs/design-notes.md` (table only) |
| + GRPO (99 steps, LoRA) | −0.569 (best@8 +0.034) | same |

The source repo also reports a mean expected Bayes ceiling of 1.722 on these patients. No
trained model cleared the blank-record floor on mean reward.

## Known limitations

- **Synthetic medicine.** The likelihoods are invented, and the Bayes baselines use the same
  generative model as the data. Strong results mean the agent learned this mapping, not that
  it can practise medicine.
- **Treatment term vs proper scoring.** The treatment terms are linear in `mass_on_truth`, so
  an agent that prescribes has a small incentive to put extra mass on conditions its
  prescription treats. The diagnosis term dominates: moving mass that way loses more Brier
  than it gains, as the exploit tests check. Still, a report followed by a prescription is
  not exactly proper.
- **Contraindications use ground-truth labs.** A penalty can fire for a lab the agent never
  ordered. The source's docstring says contraindications are "checkable by the agent", which
  is not quite true.
- **Structured output depends on the provider.** If a provider ignores `response_format`,
  format errors end episodes. Report the `decode_failure` rate alongside the reward.
- vf-eval's trace-mean headline is turn-weighted (see *Reading eval output*).

## Source and citation

Environment, generator, observation filter and reward:
[Manas-Ganti/healthcare-RL](https://github.com/Manas-Ganti/healthcare-RL), pinned at commit
`5046c92`. Packaged for verifiers 0.4.0.
