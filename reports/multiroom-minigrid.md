# Multi-room MiniGrid (source: `multiroom-minigrid/`)

**Not a git repo; not on GitHub.** Contains `venv/`, `checkpoints/` and `runs/` locally.

## 1. Task (original)
Custom MiniGrid: 1–3 rooms chained left to right. Room i holds i+1 keys. Doors consume carried
keys and open only when their counter reaches the requirement. Single-item carry, 7×7 egocentric
partial view. Discrete MiniGrid actions. `max_steps` 100/250/500 for 1/2/3 keys.

## 2. Reward (original)
Sparse and terminal: open the final door (MiniGrid-style success reward). See `envs/`. The
optimal step counts come from a scripted BFS solver (`tests/test_env.py`).

## 3. Verifiability
Programmatic success, with BFS optimal steps as a ceiling.

## 4. Reward-hacking audit (for an LLM wrapper)
Text gridworlds (BabyAI-text / BALROG-style) are an established LLM-agent benchmark. Rendering
the 7×7 view as text plus tool actions (`forward`, `left`, `right`, `pickup`, `toggle`) is
meaningful. Risks: very long horizons (115 optimal steps at 3 keys means many turns per episode,
which is expensive), and a fully observed text render would remove the memory challenge. Keep the
egocentric view. No reward exploit is apparent: success is binary and checked on env state.

## 5. Data
Procedural (seeded layouts). No license issues.

## 6. Results
`results/` has only `layouts.png` and `observation_keys3.png`. The README says "Phase 1". There are
`runs/` dirs but no curated results file, so there are **no traceable numbers**.

## 7. Hub fit verdict: **Wrappable** (new benchmark), low priority
It's legitimate as a long-horizon, memory-dependent text agent task. The main cost is turns per
episode. Since the repo isn't public, a Hub package would be the first public artifact; decide
whether you want that. Not in the default queue.
