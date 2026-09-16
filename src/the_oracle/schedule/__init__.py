"""Due work without a daemon.

PLAN.md section 9: no daemon, no queue, no scheduler process, nothing to
restart. Due work is computed on demand when the CLI starts, and one crontab
line runs the daily check. This package contains no thread and no loop.
"""

from __future__ import annotations

from the_oracle.schedule.due import (
    ACTIVITY_KINDS,
    DueSummary,
    due_summary,
    idle_days_for,
    last_activity_at,
    run_due_checks,
)

__all__ = [
    "ACTIVITY_KINDS",
    "DueSummary",
    "due_summary",
    "idle_days_for",
    "last_activity_at",
    "run_due_checks",
]
