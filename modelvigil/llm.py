"""LLM backends behind one tiny interface.

``step(system, messages, tools) -> {"text": str, "tool_calls": [...]}``

* ``AnthropicBackend`` - real tool-use with Claude (needs ANTHROPIC_API_KEY).
* ``MockBackend``      - deterministic scripted policy speaking the same
  protocol, so the full agent loop runs offline, in CI and in the benchmark.

Messages are kept in Anthropic's content-block format for both backends.
"""
from __future__ import annotations

import json
import os
from typing import Callable


class Backend:
    name = "base"

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> dict:  # pragma: no cover
        raise NotImplementedError


class MockBackend(Backend):
    """Wraps a policy ``f(history) -> (thought, tool_name, tool_input)`` where
    ``history`` is the list of (tool_name, input, result) already executed."""

    name = "mock"

    def __init__(self, policy: Callable[[list[tuple]], tuple[str, str, dict]]):
        self.policy = policy
        self._n = 0

    @staticmethod
    def _history(messages: list[dict]) -> list[tuple]:
        calls, hist = {}, []
        for m in messages:
            if not isinstance(m["content"], list):
                continue
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    calls[b["id"]] = (b["name"], b["input"])
                elif b.get("type") == "tool_result":
                    name, inp = calls[b["tool_use_id"]]
                    hist.append((name, inp, json.loads(b["content"])))
        return hist

    def step(self, system, messages, tools):
        thought, tool, args = self.policy(self._history(messages))
        self._n += 1
        return {"text": thought, "tool_calls": [{"id": f"mock_{self._n}", "name": tool, "input": args}]}


class AnthropicBackend(Backend):
    name = "anthropic"

    def __init__(self, model: str | None = None, max_tokens: int = 1500):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install anthropic  (or run with --llm mock)") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (or run with --llm mock)")
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("MODELVIGIL_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def step(self, system, messages, tools):
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                           system=system, tools=tools, messages=messages)
        text, calls = [], []
        for b in resp.content:
            if b.type == "text":
                text.append(b.text)
            elif b.type == "tool_use":
                calls.append({"id": b.id, "name": b.name, "input": dict(b.input)})
        return {"text": "\n".join(text), "tool_calls": calls}


def make_backend(kind: str, policy=None) -> Backend:
    if kind == "mock":
        return MockBackend(policy)
    if kind == "anthropic":
        return AnthropicBackend()
    raise ValueError(f"unknown LLM backend {kind!r}")
