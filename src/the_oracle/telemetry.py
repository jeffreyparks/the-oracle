"""Logfire telemetry. Optional, off unless a token is present.

The engine never depends on Logfire. ``pip install 'the-oracle[telemetry]'``
adds it; without the extra every call in this module is a cheap no-op.

What it records when enabled:

* one span per CLI command,
* one span per agent run (model, cache hit, tokens),
* every Pydantic AI model call, HTTPX request, and SQLAlchemy query.

Privacy: prompt and completion content is **not** sent by default. Set
``ORACLE_TELEMETRY_CAPTURE_CONTENT=1`` to include it, and only do that on a
project you own.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any

from the_oracle.config import Settings, get_settings

_STATUS: "TelemetryStatus | None" = None
_logfire: Any = None


@dataclass(frozen=True, slots=True)
class TelemetryStatus:
    """What happened when :func:`configure` ran. ``oracle version`` prints it."""

    requested: bool
    installed: bool
    token_present: bool
    sending: bool
    detail: str

    @property
    def active(self) -> bool:
        return self.sending

    def describe(self) -> str:
        return self.detail


def _token_present() -> bool:
    return bool(os.environ.get("LOGFIRE_TOKEN"))


def configure(settings: Settings | None = None, *, force: bool = False) -> TelemetryStatus:
    """Set up Logfire once per process. Safe to call from anywhere.

    Never raises. A broken or missing telemetry stack must not stop a lesson.
    """
    global _STATUS, _logfire
    if _STATUS is not None and not force:
        return _STATUS

    settings = settings or get_settings()
    requested = settings.telemetry

    if not requested:
        _STATUS = TelemetryStatus(False, False, _token_present(), False, "disabled by settings")
        return _STATUS

    try:
        import logfire
    except ImportError:
        _STATUS = TelemetryStatus(
            True, False, _token_present(), False,
            "logfire not installed (uv sync --extra telemetry)",
        )
        return _STATUS

    token = _token_present()
    try:
        logfire.configure(
            send_to_logfire="if-token-present",
            service_name=settings.telemetry_service_name,
            service_version=_version(),
            environment=settings.telemetry_environment or None,
            console=False,
        )
        logfire.instrument_pydantic_ai(include_content=settings.telemetry_capture_content)
        logfire.instrument_httpx(capture_headers=False)
    except Exception as exc:  # pragma: no cover - defensive
        _STATUS = TelemetryStatus(True, True, token, False, f"logfire setup failed: {exc}")
        return _STATUS

    _logfire = logfire
    _STATUS = TelemetryStatus(
        True, True, token, token,
        "sending to logfire" if token else "instrumented, no LOGFIRE_TOKEN so nothing is sent",
    )
    return _STATUS


def _version() -> str:
    from the_oracle import __version__

    return __version__


def instrument_engine(engine: Any) -> None:
    """Trace SQL for one SQLAlchemy engine. No-op when telemetry is off."""
    if _logfire is None:
        return
    try:
        _logfire.instrument_sqlalchemy(engine=engine)
    except Exception:  # pragma: no cover - defensive
        pass


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a Logfire span, or do nothing when telemetry is off."""
    if _logfire is None:
        yield None
        return
    try:
        opened = _logfire.span(name, **attributes)
    except Exception:  # pragma: no cover - defensive
        yield None
        return
    # Outside the try: an exception from the body belongs to the caller.
    with opened as active:
        yield active


def set_attributes(current: Any, **attributes: Any) -> None:
    """Add attributes to a span returned by :func:`span`. Tolerates ``None``."""
    if current is None:
        return
    try:
        for key, value in attributes.items():
            current.set_attribute(key, value)
    except Exception:  # pragma: no cover - defensive
        pass


def status() -> TelemetryStatus | None:
    """Return the last :func:`configure` result, or ``None`` if never called."""
    return _STATUS


def reset() -> None:
    """Forget the configured state. Tests use this."""
    global _STATUS, _logfire
    _STATUS = None
    _logfire = None


__all__ = [
    "TelemetryStatus",
    "configure",
    "instrument_engine",
    "reset",
    "set_attributes",
    "span",
    "status",
]
