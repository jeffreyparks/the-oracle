"""The nudge ladder and its delivery channels.

Deterministic end to end: the rung comes from idle days and due work, the body
comes from a template, and no model is ever called.
"""

from __future__ import annotations

from .channels import (
    CHANNELS,
    MemoryChannel,
    NudgeChannel,
    TerminalChannel,
    get_channel,
    register_channel,
)
from .ladder import (
    IDLE_DAYS,
    NudgeDecision,
    Rung,
    decide,
    in_quiet_hours,
    nudges_enabled,
    rung_for,
)
from .templates import TEMPLATES, due_clause, estimate_minutes, render

__all__ = [
    "CHANNELS",
    "IDLE_DAYS",
    "TEMPLATES",
    "MemoryChannel",
    "NudgeChannel",
    "NudgeDecision",
    "Rung",
    "TerminalChannel",
    "decide",
    "due_clause",
    "estimate_minutes",
    "get_channel",
    "in_quiet_hours",
    "nudges_enabled",
    "register_channel",
    "render",
]
