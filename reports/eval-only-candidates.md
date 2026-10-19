# Data/eval-only candidates (not RL environments)

## clinical_dashboard: clinical tool-calling eval (`Clinical-Intelligence-Dashboard`, public)
- `evals/cases.py`: 35 clinician questions, each paired with the expected tool call over the
  dashboard's typed tools. Graded by **exact match on the tool choice and arguments**. No judge.
  Includes "no tool exists, say so" cases.
- Data: grounded in a Synthea-generated patient DB (`clinical_synthea.db`). Synthea output is
  Apache-2.0 and synthetic, so it is redistributable, but it would need to ship as an HF dataset.
- Fit: a small, clean **tool-use eval** (Taskset + Toolset mirroring the dashboard tools).
  35 cases is too small to train on. It would pair well with the dashboard on your portfolio,
  but adds little to the Hub. Low priority.

## arcode: coding-agent eval tasks (`arcode`, public)
- 48 test-graded tasks: 34 Aider polyglot Python exercises (`poly-*`, public and possibly
  memorised; sources in `evals/tasks/NOTICE.md`), 8 hand-written multi-file tasks (`m2-*`), and
  smoke tasks. Hidden tests give the reward.
- Fit: polyglot-style coding envs already exist on the Hub. The 8 `m2-*` tasks are new but too
  few. **Skip**, unless you want a "multi-file m2" mini-suite later.

## multimodal-doc-ingestion (DocRAG-10K; not on GitHub)
- Grounded QA over 45 SEC 10-Ks with cited answers and refusals. The golden set is **not yet
  authored** and `results/` is empty. It could become a grounded-QA eval once the golden set exists. Skip for now.
