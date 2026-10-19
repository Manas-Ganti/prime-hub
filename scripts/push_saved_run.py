"""Upload a finished `vf-eval --no-push` run to the Prime Intellect Evaluations tab.

vf-eval only uploads while a run is live, and `--resume` exits before pushing when nothing
is owed. This replays the saved episodes through the same calls the runner uses
(`open_run` -> `log_episodes` -> `finish_run`), so the uploaded run is exactly the one on
disk, with no new rollouts.

    python scripts/push_saved_run.py <run_dir> [<run_dir> ...]

Needs `prime login` or $PRIME_API_KEY. Runs are uploaded private.
"""

import json
import sys
from pathlib import Path

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.episode import WireEpisode
from verifiers.v1.utils.platform import PushState, finish_run, log_episodes, open_run


def push(run_dir: Path) -> None:
    raw = json.loads((run_dir / "configs" / "resolved" / "eval.json").read_text())
    raw["push"] = True
    config = EvalConfig.model_validate(raw)
    episodes = [
        WireEpisode.model_validate(json.loads(line))
        for line in (run_dir / "traces.jsonl").read_text().splitlines()
        if line.strip()
    ]
    episodes = [e for e in episodes if e.ok]
    state = PushState()
    run = open_run(config, state, num_examples=len(episodes) // config.num_rollouts)
    if state.error or run is None or not state.url:
        sys.exit(f"{run_dir.name}: could not open a platform run ({state.error})")
    log_episodes(run, episodes)
    finish_run(run, episodes, state)
    print(f"{run_dir.name}: {len(episodes)} episodes -> {state.url}"
          + (f"  ({state.incomplete})" if state.incomplete else ""))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        push(Path(arg))
