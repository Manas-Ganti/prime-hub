# Prime Intellect Environments Hub: inventory

Workspace: `portfolio-projects/prime-hub/` (sources are sibling folders; never modified).
Built against: **verifiers `v0.4.0-38-g9350c04`** (commit `9350c041a17f`, 2026-10-17), reference
clone at `_ref/verifiers/`. The `skills/` dir has only `bump` and `release`. Use `docs/` and
`environments/` for v1 patterns.
Discovery date: 2026-10-18. Scope: `portfolio-projects/` only (per Manas). The projects named in
the brief but absent here are skipped: smart-energy-consumption-RL, the AceReason SFT subset
(only a public GitHub repo `acereasoning-sft100k` exists, no local copy), GRPO 7B scaling runs,
and the chemical plant safety monitor.

## Verdict summary

| # | Env (proposed Hub name) | Source | Verdict | Reward one-liner | Blockers | Status |
|---|---|---|---|---|---|---|
| 1 | `long-context-navigator` | `long-context-navigator-RL/` | **Native** | exact-match answer, -0.5 on ceiling/budget failure, small efficiency term | no exploit found; **needs model credentials for 5/50-task evals**; source repo now MIT (baeb433) | **Packaged** (`environments/long-context-navigator`): 50/50 tests pass (parity + exploits); vf-eval Qwen2.5-7B (vLLM bf16, n=50): id_test −0.350 ± 0.033, ood_test −0.460 ± 0.019, accuracy 0/0 |
| 2 | `dxenv-diagnostician` | `healthcare_RL/` | **Native** | severity-weighted shifted Brier − test costs − turns + treatment + predict-then-verify | coherence exploit **fixed** in v0.1 (`coherence × mass_on_truth`; +0.46 → −0.03, parity break documented); parse failures → `decode_failure` (source rule); **needs model credentials for evals**; source repo now MIT (5046c92) | **Packaged** (`environments/dxenv-diagnostician`): 80/80 tests pass (62-episode parity replay + 15 exploit tests); vf-eval Qwen2.5-7B (vLLM bf16, n=50, eval split): −0.602 ± 0.015, 94% out of turns, 0 decode failures |
| 3 | `vlm-image-forensics` | `VLM-RL-aicontent-detection/` | **Native** (not publishable yet) | ±1 verdict + 0.4 Brier(P(fake)) + 0.3 consistency − inspects − confident-wrong | **no-look exploit (+0.50)**; synth1024 not hosted; **DIV2K is academic-research-only, copyright with original owners: cannot be rehosted** (see report §5) | Phase 2 done |
| 4 | `nl-sql-chinook` | `NL-SQL-agent/` | Data/eval only | result-set equivalence vs reference SQL; refuse/clarify match | 55 items, near ceiling; verify Chinook MIT license | Phase 2 done |
| 5 | `optimal-stopping` | `finance-rl/` | Wrappable (new benchmark) | MC value of the agent's exercise boundary vs the exact Bermudan price | design work; per-step framing too noisy | Phase 2 done |
| 6 | `bond-duration-allocation` | `rl-bondportfolio/` | Wrappable, weak | MC value of the allocation rule vs the DP optimum | awkward interface | Phase 2 done |
| 7 | `multiroom-minigrid` | `multiroom-minigrid/` | Wrappable, low priority | binary success opening the final door | long horizons; repo not public | Phase 2 done |
| 8 | `clinical-tool-calling` | `clinical_dashboard/` | Data/eval only | exact match of tool call + args (35 cases) | small | Phase 2 done |
| 9 | (none) | `self-play-highwayenv/` | Not a fit | continuous control (PPO vs GRPO) | | skip |
| 10 | (none) | `maze-navigation-drone/` | Not a fit | Isaac pixel policy | | skip |
| 11 | (none) | `robotics-rl-payload-transport/` | Not a fit | Isaac continuous control | | skip |
| 12 | (none) | `arcode/` | Data/eval only, skip | hidden-test pass/fail, 48 tasks (34 Aider polyglot) | crowded on the Hub | skip |

Reports: `reports/<name>.md`. Live exploit probes: `reports/exploit_probes/` (both tests currently PASS,
meaning the exploits are live in the source reward code).

## Full project inventory

| Project | Path | What it is | Env type | Reward summary | Where results live | Git remote | Public |
|---|---|---|---|---|---|---|---|
| healthcare_RL (dxenv) | `../healthcare_RL` | LLM diagnostician over synthetic patients; Bayes ceiling | LLM-native | see #2 | `runs/phase0/results.json`, `runs/phase3/prompted_baseline.json`, `runs/grpo/steps.jsonl`; SFT/GRPO Gate B table only in `docs/design-notes.md` | Manas-Ganti/healthcare-RL | yes |
| VLM-RL-aicontent-detection | `../VLM-RL-aicontent-detection` | VLM investigates image via INSPECT cells, AI vs REAL | LLM-native (VLM) | see #3 | `results/*.md` (write-ups), W&B CSV export | RL-Based-AI-content-detector → public as Visual-reasoning-rlvr | yes |
| long-context-navigator-RL | `../long-context-navigator-RL` | multi-hop QA under a 600-token working-context ceiling | LLM-native | see #1 | model-free: `results/*.json`; LLM/SFT/GRPO: README only | Manas-Ganti/long-context-navigator | yes |
| NL-SQL-agent | `../NL-SQL-agent` | NL→SQL pipeline + 55-question eval on Chinook | not an RL env (eval) | result-set match | `results/results_{full,retrieved}.json` | Manas-Ganti/NL-SQL-agent | yes |
| finance-rl | `../finance-rl` | FQI optimal stopping for American puts vs LSM / exact | classical gym | discounted exercise payoff | `results/metrics.json`, `results/gates.json` | Manas-Ganti/optimal-stopping-rl | yes |
| rl-bondportfolio | `../rl-bondportfolio` | FQI bond duration allocation vs exact DP | classical gym | carry + duration P&L − cost − penalty | `results/metrics.json`, `results/gates.json` | Manas-Ganti/bond-duration-allocation | yes |
| multiroom-minigrid | `../multiroom-minigrid` | multi-key-door MiniGrid, PPO / RND / ICM / curriculum | classical gym | sparse terminal success | none curated (`results/` has 2 PNGs) | none (not a git repo) | no |
| self-play-highwayenv | `../self-play-highwayenv` | PPO vs critic-free GRPO on highway-env racetracks | classical gym (continuous) | driving reward (lap progress etc.) | `results/` empty | Manas-Ganti/self-play-highwayenv | yes |
| maze-navigation-drone | `../maze-navigation-drone` | Isaac Lab Crazyflie maze, vision PPO | classical sim (pixels) | geodesic potential shaping + time/collision/goal | `results/maze_checks/` (geometry only) | Manas-Ganti/maze-navigation-drone | yes |
| robotics-rl-payload-transport | `../robotics-rl-payload-transport` | Isaac payload transport vs Nav2 under OOD | classical sim | success / collision / efficiency | `results/clips/` only | Manas-Ganti/robotics-rl-payload-transport | yes |
| clinical_dashboard | `../clinical_dashboard` | full-stack clinical data platform + LLM tool calling | not an env (35-case tool-call eval) | exact tool-call match | `evals/` (no saved results file found) | Manas-Ganti/Clinical-Intelligence-Dashboard | yes |
| arcode | `../arcode` | TypeScript coding agent + 48-task eval | not an env (eval) | hidden tests | `evals/results/` | Manas-Ganti/arcode | yes |
| multimodal-doc-ingestion | `../multimodal-doc-ingestion` | grounded QA over 10-Ks | not an env (eval, no golden set) | n/a | `results/` empty | none | no |
| alignment-distillation | `../alignment-distillation` | DPO × step-distillation study on diffusion models | not an env | n/a | `results/` empty | Manas-Ganti/alignment-distillation-test | yes |
| ads-ctr | `../ads-ctr` | CTR model, batch vs streaming feature skew | not an env | n/a | none (not run on Criteo) | none | no |
| twostage-sys-rec | `../twostage-sys-rec` | two-stage recommender, honest eval protocol | not an env | n/a | `results_arc/`, `results_laptop/` | Manas-Ganti/mind-the-gap-recsys | yes |
| credit-risk | `../credit-risk` | PD model with regulatory validation | not an env | n/a | `artifacts/` | Manas-Ganti/credit-risk-model | yes |
| jobsearch-agent | `../jobsearch-agent` | daily job-posting ranking agent | not an env | n/a | n/a | Manas-Ganti/jobsearch-agent | yes |
| resume-tailor-agent | `../resume-tailor-agent` | fact-grounded resume tailoring | not an env | n/a | n/a | none | no |
| cli-test | `../cli-test` | iris ML scratch script | not an env | n/a | n/a | none | no |

## Published

| env | Hub | version | visibility | notes |
|---|---|---|---|---|
| long-context-navigator | manasganti/long-context-navigator | 0.1.2 | PUBLIC (2026-10-18) | 0.1.2 (2026-10-19) adds vf-eval baselines (ARC vLLM run) |
| dxenv-diagnostician | manasganti/dxenv-diagnostician | 0.1.2 | PUBLIC (2026-10-18) | 0.1.1 has the structured-output schema fix; 0.1.2 (2026-10-19) adds vf-eval baselines and the README output-format update |

Visibility is only set when an environment is created (`prime env push --visibility` is ignored afterwards); change it on the website.
