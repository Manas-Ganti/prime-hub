# NL-to-SQL analytics agent (source: `NL-SQL-agent/`)

Repo: `git@github.com:Manas-Ganti/NL-SQL-agent.git` (public). Last commit `530037a`.
(`.env` exists locally. It is gitignored and untracked, and it was not read.)

## 1. Task
A natural-language question over the Chinook SQLite database (11 tables). The agent pipeline is
retrieve schema → generate one SELECT, or REFUSE / CLARIFY → validate (sqlglot) → execute
read-only → synthesise. It is a product pipeline, not an RL environment. As a Hub task: one
question per episode, the agent writes SQL (optionally through an `execute_sql` tool), and the
result is graded.

## 2. Reward (as an eval)
`grade()`, `src/nl_agent/harness.py:92-117`. For query questions: result-set equivalence with the
reference SQL (`src/nl_agent/compare.py:71-93`). Order-insensitive unless the question specifies
an order; column names ignored; numerics within tol 1e-6; exact strings. For the 12
`expected: refuse|clarify` items: the status must match. Binary.

## 3. Verifiability
Programmatic: execute both queries and compare result sets. No judge.

## 4. Reward-hacking audit
- Always REFUSE: 6/55 correct (the refuse items). Always CLARIFY: 6/55. Weak, but on the Hub
  refusal should be a separate metric so it can't be farmed.
- Over-broad SELECT: fails, because the row count and column count must match.
- Reading the answer: the reference SQL is not in the prompt. With an `execute_sql` tool the
  agent can explore the DB, which is legitimate.
- Column-position sensitivity: two correct queries with swapped column order fail. That is a
  false negative, not an exploit. Document it.

## 5. Data
55 hand-written questions (`evals/questions.yaml`) + Chinook (`data/Chinook_Sqlite.sql`,
v1.4.5). The Chinook project (lerocha/chinook-database) is MIT-licensed. **Verify** before
bundling; the DB is ~1 MB, which is acceptable to vendor or build in `load()`.

## 6. Results (separate harness, measured 2026-09-14, gpt-oss-120b via VT ARC)
| | full schema | retrieved schema |
|---|---|---|
| result-match accuracy | 95% (52/55) | 85% (47/55) |
| refuse/clarify correct | 96% | 89% |
Files: `results/results_full.json`, `results/results_retrieved.json`. The README notes ±2
questions of run-to-run variance.

## 7. Hub fit verdict: **Data/eval only** (low priority)
It works mechanically, but 55 questions on one tiny schema is too small to train on, and a
frontier model is near ceiling (95%). The Hub already has larger text-to-SQL suites
(Spider/BIRD-style), so this adds little. Worth packaging only as a small "refuse/clarify-aware
SQL" eval if you want a quick, clean Hub listing.
