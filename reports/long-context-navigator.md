# Long-context navigator (source: `long-context-navigator-RL/`)

Repo: `git@github.com:Manas-Ganti/long-context-navigator.git` (public). Last commit `949b50b`.
**The repo has an active spec CLAUDE.md. It was read and not edited.**
Proposed Hub name: **`long-context-navigator`**.

## 1. Task

One instance is a ~33k-token synthetic document (configurable up to 200k) made of six
alphabetised registers plus noise sections, and a k-hop question (k = 2–3 train/ID, 4–5 OOD,
plus a scale test at 32k/64k/128k). The policy is **stateless**. Each step it sees the question,
a document map (chunk id, alphabetical range, token cost), a memory ledger, and its working
context. None of its earlier reasoning is carried forward.

Actions (`src/longctx/actions.py`, executed in `src/longctx/env.py:106+`):
`READ <id>` · `COMPRESS <ids> :: <summary>` (truncated to 120 tokens) · `DROP <ids>` · `ANSWER <value>`.

Termination: ANSWER; step budget `6 + 4·n_hops` exhausted (`env.py:126`); or **any action that
would exceed the 600-token working-context ceiling** (`env.py:141-144, 165`). Chunks are 320–480
tokens, so two chunks never fit together and compression is required.

## 2. Reward

`compute_reward`, `src/longctx/reward.py:75-88`, config in `configs/env.yaml`:

```
ceiling exceeded | budget exhausted | no answer  -> -0.5
abstain (only if allow_abstain, default off)     ->  0.1
correct                                          ->  max(0.5, 1 - 0.05 · steps_used / min_steps)
wrong                                            ->  0.0
```

Correctness is normalised exact match on alphanumerics, plus one numeric leniency: a prediction
whose only digit run equals the truth's digits matches (`reward.py:45-68`). Purely terminal.
Logged metrics (via `audit.py` / `evaluate.py`): ceiling violations, COMPRESS use, fact
retention in summaries, re-read rate, distractor capture.

## 3. Verifiability

Programmatic against generator ground truth. No judge. Every instance passes a mechanical
validator that re-reads the answer **from the text** (`validate_instance.py`): solvable along the
recorded chunks, no distractor path reaches the answer, the answer is unique, there is no lexical
shortcut, and it is solvable under the ceiling. A population-level surface-confound audit gates
generation (AUCs within 0.5 ± 0.06). A privileged oracle scores 1.000 in exactly `min_steps`.

## 4. Reward-hacking audit

No cheap exploit found.
- Guess without reading: the no-read baseline scores 0.000 (chance 0.017).
- Single-chunk shortcut: 0.183 even with the answer chunk handed over (the values are clustered).
- Multi-answer stuffing: `"4,885,866 or 4,885,870"` fails (more than one digit run). Exact
  match on normalised strings otherwise.
- Compress as a no-op: summaries are truncated to 120 tokens, and the ceiling is hard.
- Length gaming: the efficiency term only penalises steps.
- Note: a wrong answer (0.0) beats running out (-0.5), so a policy is pushed to always answer.
  That is intended, and abstention is off.
- Hub-specific risk: if the harness keeps the chat history, the model "remembers" chunks it has
  dropped, which breaks the stateless premise and the ceiling. The port must render each step as
  a fresh prompt from the env state (custom harness or per-turn prompt rebuild). **This is the
  main porting task**, and it is a correctness issue, not cosmetic.

## 5. Data

Fully procedural: `longctx generate` (seeded and byte-reproducible). The local `data/v1` is
709 MB. Don't bundle it; generate in `load()` from (split, seed). Generation includes validation
and rejection sampling, so measure the cost of generating 50–300 instances. No license problems.
Tokenizer: the default `whitespace` counter is deterministic and needs no model. Keep that on the
Hub.

## 6. Evals and ARC results (separate harness)

| measurement | setup | headline | file |
|---|---|---|---|
| mechanical baselines | 600 held-out (ID + OOD) | no-read 0.000, single-chunk oracle 0.183, full-doc oracle 1.000 | `results/baselines_mechanical.md` |
| reference policies in env | same 600 | oracle navigator 1.000; random reader 0.003 | `results/eval_{oracle,random,no-read}.md` |
| LLM full-doc (no ceiling) | Qwen2.5-7B-Instruct greedy | answer-only 0.295; reasoning 0.795 (hops 2/3/4/5: 0.944/0.931/0.761/0.532) | README (results JSON for LLM runs not in `results/`) |
| untrained 7B in env | same | 0.000 accuracy, 57% ceiling violations | README |
| SFT | LoRA r=32, 20,068 teacher rows | ID 0.630, OOD 0.050 | README |
| GRPO run 1 (lr 1e-5) | steps 20/40/60 | ID 0.653/0.600/0.623 (n.s.), KL 0.007 | README |

Only the model-free results have JSON files in `results/`. The LLM / SFT / GRPO numbers are
README-only on this Mac. Label them that way.

## 7. Hub fit verdict: **Native**

The cleanest reward in the portfolio: exact match, procedural, a validated generator, no exploit
found. A strong, distinctive Hub env (a hard context ceiling, compression as a constraint, OOD
depth generalisation). The work is the stateless harness. Knobs: split (train/id_test/ood_test/
scale_test), n_hops, doc_tokens, num_tasks, context_ceiling, seed.
