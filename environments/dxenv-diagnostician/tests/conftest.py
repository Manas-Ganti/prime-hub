import asyncio
from dataclasses import dataclass, field

import pytest
from dxenv.data.corpus import generate_corpus

from dxenv_diagnostician import taskset
from dxenv_diagnostician.taskset import DxData, DxEnv, DxEnvConfig, DxTask

FIXTURE_CORPUS = (40, 20260903)  # the source's Gate B reference sweep patients


@dataclass
class FakeTrace:
    """Just the Trace surface DxEnv.run touches."""

    last_reply: str
    prompt: str
    system_prompt: str
    ok: bool = True
    errors: list = field(default_factory=list)
    rewards: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def record_reward(self, name, value, weight=1.0):
        self.rewards[name] = value * weight

    def record_metrics(self, values):
        self.metrics.update(values)


class ScriptedAgent:
    """Replays a fixed list of completions, one per turn; repeats the last when exhausted."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.traces: list[FakeTrace] = []

    async def run(self, task):
        reply = self.replies[min(len(self.traces), len(self.replies) - 1)]
        trace = FakeTrace(reply, task.data.prompt, task.data.system_prompt)
        self.traces.append(trace)
        return trace


class FakeAgents:
    def __init__(self, diagnostician):
        self.diagnostician = diagnostician


@pytest.fixture(scope="session")
def fixture_records():
    return {r.patient_id: r for r in generate_corpus(*FIXTURE_CORPUS)}


@pytest.fixture
def play(monkeypatch, fixture_records):
    """play(patient_id, replies, budget=50, seed=0, max_turns=20) -> traces.

    Runs the real DxEnv.run against a patient from the fixture corpus (cheap) instead of
    the 20k frozen corpus."""
    monkeypatch.setattr(taskset, "_corpus", lambda: (fixture_records, {}))

    def _play(patient_id, replies, budget=50.0, seed=0, max_turns=20):
        task = DxTask(
            DxData(
                idx=0, id="t-0", split="eval", patient_id=patient_id,
                episode_seed=seed, budget=budget, max_turns=max_turns,
            )
        )
        agent = ScriptedAgent(replies)
        asyncio.run(DxEnv(DxEnvConfig(taskset={"id": "dxenv-diagnostician"})).run(task, FakeAgents(agent)))
        return agent.traces

    return _play
