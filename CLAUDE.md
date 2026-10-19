# CLAUDE.md: Prime Intellect Environments Hub prep

You are working on Manas's Mac. Your job is to find every RL environment across his project repos, understand each one's task, reward and evals well enough to explain them, decide which ones belong on the Prime Intellect Environments Hub, and turn those into Hub-ready verifiers packages with honest READMEs. You prepare; Manas publishes.

Read this whole file before touching anything.

---

## 0. Hard rules

1. **Never modify the source repos.** All Hub packaging happens in a separate workspace (`~/prime-hub/`, see section 5). Read source repos, import or vendor code from them, never edit them in place. Some repos have their own CLAUDE.md being actively used by other agents (for example the long-context navigator). Do not edit those files.
2. **Never run `prime env push` without Manas's explicit go-ahead for that specific environment.** Publishing is the one step you stop before. When he approves, push `--visibility PRIVATE` first unless he says PUBLIC.
3. **Never claim ARC numbers are Hub numbers.** Results produced on VT ARC came from his own training/eval harness. Hub-facing numbers must come from `vf-eval` / `prime eval run` on the packaged taskset. ARC results go in the README only as clearly labeled "original results (separate harness)" with the setup stated.
4. **No secrets, no private data, no big blobs** in packages. No API keys, no ARC paths, no SLURM account names, no model checkpoints, no datasets you are not licensed to redistribute.
5. **Don't read raw SLURM logs.** Manas reads those himself. Use results files (JSON/CSV/W&B exports/notebooks/paper drafts). If you need a failure explained, ask him for a curated excerpt.
6. Manas wants candid assessments. If an environment is not a good Hub fit, or its reward is hackable, say so plainly in the report rather than packaging it anyway.

---

## 1. Shared agent knowledge base (callcentre MCP)

Manas runs a central "call centre" for agents on this Mac. Before you spend time debugging anything (verifiers API errors, `uv` dependency resolution, `prime` CLI auth, eval crashes, reward bugs), check whether another agent already solved it.

Tools:
- `mcp__callcentre__search`: search solved and open issues. **Call this first** whenever you hit an error or a non-obvious decision.
- `mcp__callcentre__get`: fetch the full record for a hit.
- `mcp__callcentre__submit`: after you solve something non-trivial, record the problem, root cause and fix so the next agent doesn't re-solve it.
- `mcp__callcentre__report_outcome`: after applying a fix you found via search, report whether it worked.
- `mcp__callcentre__open_issue`: if you are stuck and can't solve it, log it as an open issue with what you tried.
- `mcp__callcentre__claim_issue`: claim an open issue before working on it so agents don't collide.
- `mcp__callcentre__log_attempt`: log each attempt on a claimed issue, including failures.
- `mcp__callcentre__resolve_issue`: close an issue you fixed, with the fix.

Good things to submit from this job: verifiers v1 migration gotchas, `pyproject.toml` patterns that the Hub accepts, reward-function pitfalls, eval-config defaults that worked.

---

## 2. What the Hub actually is (read this, it shapes every decision)

- The **Environments Hub** (app.primeintellect.ai → Environments) is a registry of installable, versioned Python packages built on Prime Intellect's **`verifiers`** library. Each package defines tasks + a harness for the model + scoring. The same package is used for evaluation, RL training (prime-rl / Hosted Training) and synthetic data.
- **It is for LLM agents.** The "policy" is a language model talking through messages and tool calls. A classical gym environment with a continuous action space and a trained MLP/CNN policy is not a Hub environment as-is.
- **Target verifiers v1** (`import verifiers.v1 as vf`): tasksets, harnesses, traces. The old v0 API (`import verifiers as vf`, `load_environment`, `SingleTurnEnv`, `vf.Rubric`) is deprecated, the Hub lists v0 packages under Legacy, and current docs say v0 has been removed. If a repo already has v0 code, port it; don't publish v0.
- The v1 docs explicitly say the repo's `skills/` directory and `AGENTS.md` are more complete than the web docs. **Before writing any v1 code:**
  1. `git clone https://github.com/PrimeIntellect-ai/verifiers.git ~/prime-hub/_ref/verifiers`
  2. Read `AGENTS.md` and everything under `skills/` (especially create/evaluate environments).
  3. Skim `https://docs.primeintellect.ai/llms.txt` and the v1 pages: tasksets, evaluation, harnesses, env.
  4. Look at real examples in `https://github.com/PrimeIntellect-ai/research-environments`.
  5. Record the exact verifiers version/commit you built against in `~/prime-hub/HUB_INVENTORY.md`. The API moves fast; when docs and installed source disagree, the installed source wins.
  If Context7 MCP tools are available, use them for library docs too.

### v1 mental model (verify against the installed version)

- **TaskData**: one immutable row (prompt, reference answer, files, timeouts). Set `id` to a durable source ID.
- **Task**: behavior for a row. Rewards are methods decorated `@vf.reward` that take a `vf.Trace` and return a float. Also `@vf.metric` (logged, not trained on) and `@vf.stop` (termination conditions). Judges via `vf.Judge` with model set in a `TaskConfig`.
- **Taskset[TaskT, ConfigT]**: `load()` returns or yields Tasks. Configurable knobs live on `TasksetConfig` (load-time) and `TaskConfig` (execution/scoring); they become CLI flags like `--env.taskset.num-tasks`. Procedural/infinite tasksets set `INFINITE = True`.
- **Toolset**: tools exposed to the harness over MCP (`@vf.tool`). Use for environments where the agent takes actions (zoom, read chunk, order lab test, query DB).
- **Harness**: the program the model runs in. Prefer built-ins; only write a custom `vf.Harness` if the interaction genuinely needs it.
- **Env**: multi-agent control flow (relevant for self-play).
- **Trace**: message graph + rewards + metrics + per-call records.

Shape of a package (from `uv run vf-init <name>`):
```
environments/<name>/
├── pyproject.toml
├── README.md               # rendered on the Hub page
└── <name>/
    ├── __init__.py         # exports the Taskset class
    └── taskset.py
```
Scaffold flags: `-T` adds a Toolset server, `-H` adds a custom harness.

### Commands
```bash
uv tool install prime && prime login            # Manas must have a username set on his profile
uv run vf-init <name>                           # scaffold
uv run vf-eval <name> -n 5                      # smoke test
uv run vf-eval <name> -n 50 ...                 # real eval (check current flags for model/rollouts/save)
prime eval run <name> -m <model> -n 5           # Prime CLI eval path
prime eval view                                 # inspect saved runs
prime env push --visibility PRIVATE             # ONLY after Manas approves
prime env push --auto-bump                      # later versions
prime env info <owner>/<name>; prime env version list <owner>/<name>
```

### pyproject gotchas
- Git dependencies must be inline PEP 508 in `[project].dependencies` (`pkg @ git+https://...`). `[tool.uv.sources]` is **not** carried into the wheel and breaks Hub installs.
- Pin `verifiers` to the version you tested.
- Fill `description` and tags; they show on the Hub.
- Keep heavy deps optional or justified. A Hub user should be able to `prime env install` and run a 5-task eval on a laptop with an API model.

---

## 3. Phase 1: Discover (read-only)

Find every project that contains an RL environment. Search `~` (skip `Library`, caches, `node_modules`, `.venv`, conda envs) for signals:
- `gymnasium`/`gym.Env`, `pettingzoo`, `highway_env`, `step(`/`reset(` pairs, `verifiers`, `load_environment`, `vf.`, `Taskset`, `reward`, `rubric`, `GRPO`, `trl`, `openrlhf`, `prime-rl`, `CLAUDE.md`, `results/`, `*.jsonl` eval outputs, `wandb/`, paper drafts (`*.tex`).

Projects Manas has that likely contain environments (use as a checklist, not a limit; confirm paths yourself):
- Synthea healthcare diagnostician env ("dxenv"/synthea): LLM diagnostician, Brier-score terminal reward with potential-based shaping, computable Bayes-optimal ceiling, pre-registered feasibility gates.
- VLM+RL AI-generated image detector ("forensics"): POMDP with active sensing, zoom/metadata actions, grounding reward verified by an intervention test. Shares predict-then-verify reward code with Synthea.
- Long-context navigator: multi-hop QA over documents larger than the context window; agent chooses what to read, compress, drop. **Has an active spec CLAUDE.md, do not edit.**
- NL-to-SQL analytics agent.
- Competitive self-play racetrack (PettingZoo + highway-env, ELO).
- Drone navigation (3D maze speed navigation).
- Robotics payload variation under distribution shift.
- Smart energy consumption RL.
- Optimal stopping for American options; duration-aware bond allocation.
- Possibly reusable as eval data rather than envs: reasoning SFT assignment (AceReason subset), GRPO 7B scaling runs, chemical plant safety monitor.

Write `~/prime-hub/HUB_INVENTORY.md` with one row per project: path, what it is, env type (LLM-native / classical gym / not an env), reward summary, where ARC results live, git remote + whether it is public.

---

## 4. Phase 2: Understand each environment

For each environment, write `~/prime-hub/reports/<name>.md` covering:

1. **Task**: what one episode is, what the agent observes, what actions exist, episode termination.
2. **Reward**: the exact formula, traced to the file and function (`path:line`). Terminal vs shaped, ranges, weights, any potential-based shaping and whether it preserves the optimal policy. Note which signals are rewards vs logged metrics.
3. **Verifiability**: is the reward computed programmatically from ground truth, by a judge model, or by a learned model? What makes it trustworthy (e.g. intervention test, Bayes-optimal ceiling)?
4. **Reward-hacking audit**: try to break it on paper. Format exploits, degenerate always-same answers, reading the answer from the prompt or tool outputs, length gaming, judge flattery. If you can construct a cheap exploit, write a tiny test that demonstrates it.
5. **Data**: source, size, splits, license, whether it can be redistributed or must be generated/downloaded at load time.
6. **Evals and ARC results**: what was measured, on which model(s), with what config, where the numbers live. Summarize the headline numbers and the exact setup. Do not re-derive numbers you can't trace to a file.
7. **Hub fit verdict**: one of
   - **Native**: already an LLM-agent environment; port to a v1 taskset (+ Toolset if it has actions).
   - **Wrappable**: classical env where a text/tool interface for an LLM is meaningful. Be explicit that this creates a *new* benchmark; the original trained-policy results do not transfer.
   - **Not a fit**: continuous-control or pixel-policy work where an LLM wrapper would be contrived. Say so and skip.
   - **Data/eval only**: better published as an eval taskset than a training env.

Expected rough sort (confirm from the code, don't trust this): Synthea, VLM forensics, long-context navigator, NL-to-SQL are Native. Optimal stopping, bond allocation, smart energy are possibly Wrappable as sequential decision tasks with tool actions. Drone, payload, racetrack are probably Not a fit (racetrack self-play could be wrappable via a multi-agent `Env` only if decisions are coarse enough to be meaningful as text; argue it either way in the report).

Stop after Phase 2 and send Manas the inventory + verdicts as structured options (he decides incrementally). Ask which environments to package and in what order. Default suggestion: Synthea first (cleanest verifiable reward, clear ceiling), then forensics (reuses the predict-then-verify code), then long-context navigator, then NL-to-SQL.

---

## 5. Phase 3: Package (only for environments Manas approves)

Workspace:
```
~/prime-hub/
├── HUB_INVENTORY.md
├── reports/<name>.md
├── _ref/verifiers/            # reference clone, never edited
└── environments/<name>/       # one Hub package each
```

For each approved environment:
1. `uv run vf-init <name>` in `~/prime-hub/environments/` (add `-T` if the agent takes actions). Hub names: lowercase, hyphenated, descriptive (`synthea-diagnostician`, not `dxenv-v3-final`).
2. Port logic into `taskset.py`: TaskData fields, reward functions as `@vf.reward` methods with the **same math** as the original, auxiliary signals as `@vf.metric`, termination as `@vf.stop`. Expose real knobs (split, difficulty, num tasks, judge model) on configs; nothing else.
3. Shared reward code (predict-then-verify between Synthea and forensics): put it in a small shared module both packages depend on, or vendor it into each. Don't depend on a local path.
4. Data: load from Hugging Face or generate procedurally inside `load()`. Never bundle large files. If the data can't be redistributed, the package must fetch or generate it, or the env is not publishable; flag it.
5. **Reward parity test**: write `tests/test_reward_parity.py` that feeds fixed trajectories/answers through both the original reward code and the packaged `@vf.reward` and asserts equal outputs. This is how we know the Hub env measures the same thing as the ARC work.
6. **Exploit tests**: turn every exploit from the Phase 2 audit into a test asserting it scores low.
7. Smoke eval: `uv run vf-eval <name> -n 5` with a cheap API model. Then a 50-task eval. Save outputs. If the reward is always 0 or always 1, stop and investigate before going further.
8. README.md (this is the Hub page). Sections: overview; task and interaction; reward (formula, ranges, what's a metric vs reward); data and license; configuration knobs; quickstart (`prime env install`, `vf-eval` command); baseline results from `vf-eval` (model, n, rollouts, mean ± spread); **original results** from ARC, clearly labeled with their setup and the note that they used a separate harness; known limitations; citation (link the paper/portfolio site if public). Keep it plain and precise.
9. `pyproject.toml`: pinned verifiers, inline git deps, description, tags, version `0.1.0`.
10. Run `prime env push` dry checks if the CLI offers them; otherwise stop here.
11. Update `HUB_INVENTORY.md` status and append a "ready to push" entry with the exact push command. Then hand over to Manas.

---

## 6. Definition of done (per environment)

- [ ] Report written with reward traced to source lines and a fit verdict
- [ ] v1 package scaffolded in `~/prime-hub/environments/<name>`
- [ ] Reward parity tests pass
- [ ] Exploit tests pass
- [ ] 5-task and 50-task evals run, outputs saved, numbers in README
- [ ] ARC results included only with clear labeling
- [ ] No secrets, ARC paths, checkpoints or non-redistributable data
- [ ] Git deps inline, verifiers pinned
- [ ] Anything non-trivial you solved submitted to the callcentre
- [ ] Manas approved before any push; first push PRIVATE

---

## 7. How to report back

Short and candid. For each environment: verdict, one line on the reward, test status, eval numbers, blockers. Present decisions as numbered options with your recommendation first. Flag anything that would embarrass a public release (hackable reward, unreproducible numbers, license problem) before anything else.