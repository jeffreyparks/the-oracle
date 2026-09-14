"""Record once, replay forever. The agent test harness.

Agent tests must run with no network and no API key, or agent testing dies
within two weeks (PLAN.md section 9). So every LLM call inside a test goes
through here.

* **Replay** (the default, and what CI does): look the prompt up in a JSON
  cassette and return the stored output. A miss is a loud failure.
* **Synthesize**: opt-in per test. A local, deterministic responder stands in
  for the model. Used where the prompt depends on a component another worker
  owns, so a stale key cannot turn into a network call.
* **Record**: set ``ORACLE_CASSETTE_RECORD=1`` with a real API key. Misses hit
  the live model once and are written to the cassette.

The key is ``sha256(agent name + prompt)``, so a prompt change invalidates
exactly the entries it should.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from the_oracle.agents.base import Agent, Usage

CASSETTE_DIR = Path(__file__).resolve().parent
RECORD_ENV = "ORACLE_CASSETTE_RECORD"

#: ``(agent_name, prompt) -> output dict``. A stand-in for the model.
Synthesizer = Callable[[str, str], dict[str, Any]]


class CassetteMiss(AssertionError):
    """Raised when a prompt is not in the cassette and recording is off."""


def key_for(agent_name: str, prompt: str) -> str:
    """Stable key for one agent call."""
    digest = hashlib.sha256(f"{agent_name}\n{prompt}".encode())
    return digest.hexdigest()[:32]


def recording() -> bool:
    """True when the suite is allowed to talk to a real model."""
    return os.environ.get(RECORD_ENV, "") not in ("", "0", "false")


class Cassette:
    """One JSON file of recorded agent calls."""

    def __init__(
        self,
        path: Path,
        *,
        record: bool | None = None,
        synthesize: Synthesizer | None = None,
    ) -> None:
        self.path = path
        self.record = recording() if record is None else record
        self.synthesize = synthesize
        self.entries: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []
        self._dirty = False
        if path.is_file():
            self.entries = json.loads(path.read_text(encoding="utf-8"))["entries"]

    # -- storage -----------------------------------------------------------

    def save(self) -> None:
        """Write the cassette back if anything new landed in it."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {"version": 1, "entries": self.entries}
        self.path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._dirty = False

    def put(self, agent_name: str, prompt: str, output: dict[str, Any], usage: Usage) -> None:
        self.entries[key_for(agent_name, prompt)] = {
            "agent": agent_name,
            "prompt": prompt,
            "output": output,
            "usage": {
                "request_tokens": usage.request_tokens,
                "response_tokens": usage.response_tokens,
            },
        }
        self._dirty = True

    # -- playback ----------------------------------------------------------

    async def play(self, agent: Agent[Any, Any], prompt: str) -> tuple[Any, Usage]:
        """Return the stored output for ``prompt``, or fill the gap."""
        name = type(agent).name
        entry = self.entries.get(key_for(name, prompt))
        source = "cassette"
        if entry is None:
            entry, source = await self._fill(agent, name, prompt)
        self.calls.append({"agent": name, "prompt": prompt, "source": source})
        usage_data = entry.get("usage", {})
        usage = Usage(
            request_tokens=int(usage_data.get("request_tokens", 0)),
            response_tokens=int(usage_data.get("response_tokens", 0)),
        )
        return type(agent).output_type.model_validate(entry["output"]), usage

    async def _fill(
        self, agent: Agent[Any, Any], name: str, prompt: str
    ) -> tuple[dict[str, Any], str]:
        if self.record:
            run = await agent.pydantic_agent().run(prompt)
            usage = Usage.from_run(getattr(run, "usage", None))
            self.put(name, prompt, run.output.model_dump(mode="json"), usage)
            return self.entries[key_for(name, prompt)], "live"
        if self.synthesize is not None:
            output = self.synthesize(name, prompt)
            self.put(name, prompt, output, Usage(0, 0))
            return self.entries[key_for(name, prompt)], "synthesized"
        raise CassetteMiss(
            f"no cassette entry for {name} in {self.path.name}.\n"
            f"prompt was:\n{prompt}\n"
            f"re-record with {RECORD_ENV}=1 and a real API key."
        )


@contextmanager
def cassette(
    name: str,
    *,
    synthesize: Synthesizer | None = None,
    write: bool | None = None,
) -> Iterator[Cassette]:
    """Patch every agent LLM call onto the cassette named ``name``.

    ``write`` forces the cassette to be saved on exit; by default it is saved
    only while recording.
    """
    tape = Cassette(CASSETTE_DIR / f"{name}.json", synthesize=synthesize)

    async def _fake_run_llm(self: Agent[Any, Any], prompt: str) -> tuple[Any, Usage]:
        return await tape.play(self, prompt)

    with patch.object(Agent, "_run_llm", _fake_run_llm):
        yield tape

    if write or (write is None and tape.record):
        tape.save()
