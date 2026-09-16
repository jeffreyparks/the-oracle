"""The one crontab line, and the safe way to install it.

There is no scheduler process here. The whole "daemon" is a single line in the
user's crontab that runs ``oracle cron run`` once a day. Everything in this
module is about editing that one line honestly:

* the exact change is computed before anything is written;
* only our line, marked with :data:`MARKER`, is ever removed;
* the command that talks to ``crontab(1)`` is injected, so a test can never
  touch a real crontab. :class:`SystemCrontab` additionally refuses to write
  while pytest is running.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

#: Every line we manage ends with this. Nothing else is ever touched.
#: Kept as ``the-oracle`` even though the command is now ``oracle``: this string
#: identifies lines already installed in a user's crontab, and changing it would
#: orphan them so ``cron uninstall`` could never find them again.
MARKER: str = "# the-oracle:nudge"

#: What the crontab line runs. Kept here so tests and docs cannot drift.
SUBCOMMAND: str = "cron run"

DEFAULT_HOUR: int = 9
DEFAULT_MINUTE: int = 0

STAMP_NAME: str = "cron-last-run.json"


class CrontabError(RuntimeError):
    """Raised when the crontab cannot be read or written."""


@runtime_checkable
class CrontabRunner(Protocol):
    """The seam. Two calls, both trivially fakeable."""

    def read(self) -> str: ...

    def write(self, text: str) -> None: ...


class SystemCrontab:
    """The real ``crontab(1)``. The only class in this package that shells out."""

    def __init__(self, executable: str = "crontab") -> None:
        self.executable = executable

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def read(self) -> str:
        if not self.available():
            raise CrontabError(f"{self.executable} was not found on PATH")
        result = subprocess.run(  # noqa: S603
            [self.executable, "-l"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0 and "no crontab" not in (result.stderr or "").lower():
            raise CrontabError((result.stderr or "crontab -l failed").strip())
        return result.stdout or ""

    def write(self, text: str) -> None:
        self._refuse_under_test()
        if not self.available():
            raise CrontabError(f"{self.executable} was not found on PATH")
        payload = text if text.endswith("\n") or not text else text + "\n"
        result = subprocess.run(  # noqa: S603
            [self.executable, "-"], input=payload, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise CrontabError((result.stderr or "crontab write failed").strip())

    @staticmethod
    def _refuse_under_test() -> None:
        """A test that writes to a developer's real crontab is a bug, not a test."""
        if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
            raise CrontabError(
                "refusing to write a real crontab from a test; inject a CrontabRunner"
            )


class MemoryCrontab:
    """An in-memory crontab. The default under pytest and the fixture for tests."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.writes: list[str] = []

    def read(self) -> str:
        return self.text

    def write(self, text: str) -> None:
        self.text = text if text.endswith("\n") or not text else text + "\n"
        self.writes.append(self.text)


_RUNNER: CrontabRunner | None = None


def get_runner() -> CrontabRunner:
    """The runner in force. Under pytest this is an in-memory crontab."""
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = MemoryCrontab() if "pytest" in sys.modules else SystemCrontab()
    return _RUNNER


def set_runner(runner: CrontabRunner | None) -> None:
    """Install a runner. Tests use this; nothing in ``src`` calls it."""
    global _RUNNER
    _RUNNER = runner


# -- the line ----------------------------------------------------------------


def executable_path() -> str:
    """The ``oracle`` entry point to call, absolute where we can find it."""
    found = shutil.which("oracle") or shutil.which("the-oracle")
    if found:
        return found
    guess = Path(sys.executable).with_name("oracle")
    if not guess.exists():
        guess = Path(sys.executable).with_name("the-oracle")
    if guess.exists():
        return str(guess)
    return "oracle"


def cron_line(
    *,
    hour: int = DEFAULT_HOUR,
    minute: int = DEFAULT_MINUTE,
    executable: str | None = None,
) -> str:
    """The exact line we install. Printing it is a complete alternative to install."""
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    if not 0 <= minute <= 59:
        raise ValueError("minute must be between 0 and 59")
    exe = executable or executable_path()
    return f"{minute} {hour} * * * {exe} {SUBCOMMAND} >/dev/null 2>&1 {MARKER}"


def is_ours(line: str) -> bool:
    return MARKER in line


def find_ours(text: str) -> list[str]:
    """Every managed line currently in the crontab."""
    return [line for line in text.splitlines() if is_ours(line)]


@dataclass(frozen=True, slots=True)
class CrontabChange:
    """A proposed edit, computed before anything is written."""

    action: str  # "install" | "uninstall"
    changed: bool
    reason: str
    before: str
    after: str
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()

    def diff(self) -> str:
        """The exact change, in the plainest form there is."""
        lines = [f"- {line}" for line in self.removed] + [f"+ {line}" for line in self.added]
        return "\n".join(lines) or "(no change)"


def _joined(lines: list[str]) -> str:
    return ("\n".join(lines) + "\n") if lines else ""


def plan_install(
    runner: CrontabRunner,
    *,
    hour: int = DEFAULT_HOUR,
    minute: int = DEFAULT_MINUTE,
    executable: str | None = None,
) -> CrontabChange:
    """Work out the edit without making it. ``changed`` is False on a duplicate."""
    before = runner.read()
    line = cron_line(hour=hour, minute=minute, executable=executable)
    existing = find_ours(before)

    if line in existing:
        return CrontabChange(
            action="install",
            changed=False,
            reason="that exact line is already installed",
            before=before,
            after=before,
        )

    kept = [entry for entry in before.splitlines() if not is_ours(entry)]
    if existing:
        # One managed line, always. A changed hour replaces, never duplicates.
        return CrontabChange(
            action="install",
            changed=True,
            reason="replacing the existing oracle line",
            before=before,
            after=_joined([*kept, line]),
            added=(line,),
            removed=tuple(existing),
        )
    return CrontabChange(
        action="install",
        changed=True,
        reason="adding the oracle line",
        before=before,
        after=_joined([*kept, line]),
        added=(line,),
    )


def plan_uninstall(runner: CrontabRunner) -> CrontabChange:
    """Remove only the marked line. Every other entry is preserved verbatim."""
    before = runner.read()
    existing = find_ours(before)
    if not existing:
        return CrontabChange(
            action="uninstall",
            changed=False,
            reason="no oracle line is installed",
            before=before,
            after=before,
        )
    kept = [entry for entry in before.splitlines() if not is_ours(entry)]
    return CrontabChange(
        action="uninstall",
        changed=True,
        reason="removing the oracle line",
        before=before,
        after=_joined(kept),
        removed=tuple(existing),
    )


def apply(change: CrontabChange, runner: CrontabRunner) -> bool:
    """Write a planned change. Returns whether anything was written."""
    if not change.changed:
        return False
    runner.write(change.after)
    return True


# -- status ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CronStatus:
    installed: bool
    lines: tuple[str, ...]
    last_run_at: datetime | None
    last_result: str | None


def stamp_path() -> Path:
    from the_oracle.config import get_settings

    return Path(get_settings().home) / STAMP_NAME


def record_run(*, at: datetime, result: str) -> Path:
    """Note that the cron line ran. ``cron status`` reads this back."""
    path = stamp_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"last_run_at": at.isoformat(), "result": result}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_stamp() -> tuple[datetime | None, str | None]:
    path = stamp_path()
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("last_run_at")
        moment = datetime.fromisoformat(raw) if raw else None
        if moment is not None and moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment, data.get("result")
    except (OSError, ValueError):
        return None, None


def status(runner: CrontabRunner | None = None) -> CronStatus:
    runner = runner or get_runner()
    lines = find_ours(runner.read())
    last_run_at, result = read_stamp()
    return CronStatus(
        installed=bool(lines),
        lines=tuple(lines),
        last_run_at=last_run_at,
        last_result=result,
    )


__all__ = [
    "DEFAULT_HOUR",
    "MARKER",
    "SUBCOMMAND",
    "CronStatus",
    "CrontabChange",
    "CrontabError",
    "CrontabRunner",
    "MemoryCrontab",
    "SystemCrontab",
    "apply",
    "cron_line",
    "executable_path",
    "find_ours",
    "get_runner",
    "is_ours",
    "plan_install",
    "plan_uninstall",
    "read_stamp",
    "record_run",
    "set_runner",
    "stamp_path",
    "status",
]
