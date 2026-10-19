# long-context-navigator

Multi-hop question answering over a document far larger than the context the agent may hold.
The agent decides what to read, what to keep, what to compress and what to drop. It works under
a **hard working-context ceiling**, and any action that would exceed the ceiling ends the
episode as a failure. Compression is a constraint to satisfy, not a cost to trade off.

The question is whether an agent learns a compression policy that carries every bridge entity
across reads and generalises to deeper chains than it was trained on, or whether it finds one
lucky chunk and guesses. The generator, validator and reward are built so the answer can be
trusted in either direction.

## Task and interaction

An instance is a ~32k-token synthetic document (64k and 128k in `scale_test`). It is made of
six alphabetised registers (projects, personnel, departments, facilities, regional offices,
vendors) plus irrelevant noise sections. The question follows a k-hop chain:

> *What is the annual budget of the unit of the lead of project Amber Falcon?*
> project → **Lead** (person) → **Unit** (department) → **Annual budget**

Only the anchor and the final attribute are named. Every bridge entity has to be discovered by
reading. The working-context ceiling is **600 tokens** and chunks are 320–480 tokens, so two
chunks never fit together: each bridge has to be written into a summary before its chunk is
freed. Every instance includes distractors: near-miss entities (`Amber Falcon II`, sorted next to
the real one), superseded statements ("previously …"), and near-miss values clustered within a
few percent of the answer.

**The policy is stateless.** Each step is a fresh single-turn model call whose prompt is the
rendered state: question, document map (chunk id, alphabetical range, token cost), memory ledger,
and working context. No chat history is carried between steps, so a COMPRESS summary is the
only memory. Each step is recorded as its own trace.

One action per step, on the last line:

| action | effect |
|---|---|
| `ACTION: READ <id>` | load a chunk (its token count is charged against the ceiling) |
| `ACTION: COMPRESS <ids> :: <summary>` | replace held items with your own summary (truncated at 120 tokens) |
| `ACTION: DROP <ids>` | free held items; re-reading later is allowed and costs a step |
| `ACTION: ANSWER <value>` | terminal |

The episode ends on ANSWER, when the step budget (`6 + 4·n_hops`) runs out, or as soon as an
action would push the working context above 600 tokens.

## Reward

One terminal reward per episode, recorded on **every step trace of that episode**:

```
ceiling exceeded | step budget exhausted | no answer  ->  -0.5
wrong answer                                         ->   0.0
correct answer                                       ->   max(0.5, 1 - 0.05 · steps_used / min_steps)
```

`min_steps` is the instance's provably minimal step count, computed by the generator, so deep
instances are not penalised for being deep. Correctness is normalised exact match on
alphanumerics, plus one numeric leniency: a prediction whose only digit run equals the truth's
digits matches (`USD 4,885,866` matches `4,885,866`; `4,885,866 or 4,885,870` does not). There
is no judge.

Metrics (logged, not rewarded): `correct`, `ceiling_exceeded`, `budget_exhausted`,
`steps_over_min`, `n_compress`, `n_invalid`, `distractor_capture` (the wrong answer is a known
distractor's value), and `final_step` (1.0 on the episode's last trace).

**Reading eval output.** vf-eval's headline is a mean over *traces*, and an episode has one trace
per step, so longer episodes are weighted more heavily. For per-episode numbers, run
`python scripts/episode_summary.py <run_dir>`. For training, every step trace carries its
episode's return, so one advantage per episode is inherited by all of its steps (as in the
original GRPO loop). A consumer that computes advantages per trace group should group by episode.

## Why the reward can be trusted

Every instance passes a mechanical validator before use. The validator re-reads the answer
**from the text**, not from generator state, and checks that:
- the chain is solvable along exactly the recorded chunks;
- no distractor path reaches the answer;
- the answer value is unique;
- the anchor never appears in the answer chunk (no lexical shortcut);
- the instance is solvable under the ceiling.

A population-level audit rejects surface confounds (chunk position, length and header must not
predict evidence; AUC within 0.5 ± 0.06). The audit in the source repo measured:
- a privileged oracle navigator scores 1.000 in exactly `min_steps`;
- a no-read policy scores 0.000 (chance is 0.017);
- the single-chunk shortcut caps at 0.183 even when the answer chunk is handed over.

`tests/test_exploits.py` checks that none of these strategies score above 0.0: guessing without
reading, answering "unknown", listing every candidate value, concatenating digits, reading two
chunks without compressing, copying a chunk into a "summary", spamming invalid actions, and
stalling with re-reads.

## Data and license

Fully procedural. Instances are generated in `load()`/`run()` from (split, index) by the source
repo's seeded generator, so nothing is downloaded or bundled. The generated instances are
identical to the source repo's `data/v1` splits (checked by `tests/test_reward_parity.py`).

| split | n | hops | doc tokens |
|---|---|---|---|
| `train` | 2000 | 2–3 | 32k |
| `id_test` (default) | 300 | 2–3 | 32k |
| `ood_test` | 300 | 4–5 | 32k |
| `scale_test` | 150 | 3 | 32k / 64k / 128k |

Token counts use the source repo's deterministic whitespace/punctuation counter, not any model's
tokenizer.

## Configuration

| flag | default | meaning |
|---|---|---|
| `--env.taskset.split` | `id_test` | which split to generate |
| `--env.taskset.num-tasks` | split size | generate more (or fewer) instances; extra ones use fresh seeds |
| `--env.navigator.model` / `.sampling.*` | run's model | the navigating model |

The ceiling, step budget and reward weights are fixed to the source environment's values, so
results stay comparable to the original work.

## Quickstart

```bash
prime env install manasganti/long-context-navigator
uv run vf-eval long-context-navigator -n 5 -r 1 -m <model>
uv run vf-eval long-context-navigator --env.taskset.split ood_test -n 50 -r 1 -m <model>
python scripts/episode_summary.py outputs/long-context-navigator--<model>--null--<id>
```

Each step is one tool-less model call (`null` harness, local `subprocess` runtime). A 5-task
smoke eval runs on a laptop with any OpenAI-compatible API model. Keep `max_tokens` around 512
per step, since a step needs only a THOUGHT line and an ACTION line.

## Baseline results (vf-eval, this package)

Per-episode means from `scripts/episode_summary.py`. The run: `vf-eval -n 50 -r 1`, with the
source repo's rollout sampling (T = 0.8, top_p = 0.95, 256 max tokens per step). The model was
served unquantized (bf16) by vLLM 0.8.5 on one A100. Each split took under 2 minutes.

| model | split | episodes | reward (mean ± SE) | accuracy | ceiling exceeded | step budget exhausted | wrong answer |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | `id_test` (2–3 hops) | 50 | −0.350 ± 0.033 | 0.00 | 66% | 4% | 30% |
| Qwen2.5-7B-Instruct | `ood_test` (4–5 hops) | 50 | −0.460 ± 0.019 | 0.00 | 90% | 2% | 8% |

The untrained 7B model never answers correctly. It mostly fails by loading a second chunk
before compressing the first, which breaks the 600-token ceiling. It compressed in only 8%
(`id_test`) and 2% (`ood_test`) of episodes. This matches the original work's
in-environment result for the same model (0.000 on both splits, below).

## Original results (separate harness)

These numbers come from the source repo's own training/eval harness (Qwen2.5-7B-Instruct, VT ARC
A100s, 600 held-out instances: 300 ID + 300 OOD). They were **not** produced by this package or
by vf-eval, and they are not Hub baselines. The model-free rows are in the source repo's
`results/`. The LLM rows are reported in its README.

| policy | ID (2–3 hops) | OOD (4–5 hops) |
|---|---|---|
| privileged oracle navigator | 1.000 | 1.000 |
| random reader / no-read | 0.000–0.003 | 0.000–0.007 |
| Qwen2.5-7B-Instruct, untrained, in-environment | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct + SFT (LoRA r=32, 20k oracle-teacher steps) | 0.630 | 0.050 |
| SFT + GRPO (lr 1e-5, steps 20 / 40 / 60) | 0.653 / 0.600 / 0.623 | 0.077 / 0.053 / 0.060 |
| reference: full document in view, no ceiling, reasoning | 0.936 | 0.653 |

The first GRPO run changed nothing measurable (all differences within ~2 SE). The source repo
keeps it as a control.

## Known limitations

- Synthetic registers, not natural text. A MuSiQue substrate was tried and rejected in the source
  repo (ceiling 0.374).
- The 600-token ceiling is enforced by the environment, not by the model's context window. The
  model never sees more than one rendered state per call.
- vf-eval's trace-mean headline is step-weighted (see *Reading eval output*).
- Per-step overhead: every step is a separate rollout (about 1–2 s of harness overhead locally on
  top of the model call).

## Source and citation

Environment, generator, validator and reward: [Manas-Ganti/long-context-navigator](https://github.com/Manas-Ganti/long-context-navigator)
(pinned at commit `baeb433`). Packaged for verifiers 0.4.0.
