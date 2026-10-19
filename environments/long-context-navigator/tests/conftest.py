import asyncio
from dataclasses import dataclass, field

import pytest

from long_context_navigator.taskset import (
    NavigatorData,
    NavigatorEnv,
    NavigatorEnvConfig,
    NavigatorTask,
    split_params,
)


@dataclass
class FakeTrace:
    """Just the Trace surface NavigatorEnv.run touches."""

    last_reply: str
    prompt: str
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
    """Replays a fixed list of completions, one per step; repeats the last when exhausted."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.traces: list[FakeTrace] = []

    async def run(self, task):
        reply = self.replies[min(len(self.traces), len(self.replies) - 1)]
        trace = FakeTrace(last_reply=reply, prompt=task.data.prompt)
        self.traces.append(trace)
        return trace


class FakeAgents:
    def __init__(self, navigator):
        self.navigator = navigator


def task_for(split: str, i: int) -> NavigatorTask:
    seed, n_hops, doc_tokens = split_params(split, i)
    return NavigatorTask(
        NavigatorData(
            idx=i, id=f"{split}-{i}", split=split, seed=seed, n_hops=n_hops, doc_tokens=doc_tokens
        )
    )


def play(split: str, i: int, replies) -> list[FakeTrace]:
    env = NavigatorEnv(NavigatorEnvConfig(taskset={"id": "long-context-navigator"}))
    agent = ScriptedAgent(replies)
    asyncio.run(env.run(task_for(split, i), FakeAgents(agent)))
    return agent.traces


@pytest.fixture
def play_episode():
    return play
