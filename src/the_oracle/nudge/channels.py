"""Delivery channels. The protocol is the point; terminal is the only adapter.

Email and chat arrive later as new classes registered here. Nothing else in the
system learns how a nudge is delivered.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rich.console import Console
from rich.panel import Panel

from the_oracle.store.models import Nudge

__all__ = [
    "CHANNELS",
    "MemoryChannel",
    "NudgeChannel",
    "TerminalChannel",
    "get_channel",
    "register_channel",
]


@runtime_checkable
class NudgeChannel(Protocol):
    """Anything that can put a nudge in front of a learner."""

    name: str

    def send(self, nudge: Nudge) -> bool:
        """Deliver ``nudge``. Return True when it reached the learner."""
        ...


class TerminalChannel:
    """Write the nudge to stdout with Rich. Quiet borders, no decoration."""

    name = "terminal"

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def send(self, nudge: Nudge) -> bool:
        body = (nudge.body or "").strip()
        if not body:
            return False
        self.console.print(Panel(body, title="the oracle", border_style="cyan"))
        return True


class MemoryChannel:
    """Collect nudges in a list. For tests and dry runs."""

    name = "memory"

    def __init__(self) -> None:
        self.sent: list[Nudge] = []

    def send(self, nudge: Nudge) -> bool:
        if not (nudge.body or "").strip():
            return False
        self.sent.append(nudge)
        return True


CHANNELS: dict[str, type] = {
    TerminalChannel.name: TerminalChannel,
    MemoryChannel.name: MemoryChannel,
}


def register_channel(name: str, factory: type) -> None:
    """Add a channel implementation under ``name``."""
    CHANNELS[name] = factory


def get_channel(name: str = "terminal") -> NudgeChannel:
    """Return a channel by name. Unknown names fail loud."""
    try:
        factory = CHANNELS[name]
    except KeyError:
        known = ", ".join(sorted(CHANNELS))
        raise ValueError(f"unknown nudge channel {name!r}. known: {known}") from None
    return factory()  # type: ignore[return-value]
