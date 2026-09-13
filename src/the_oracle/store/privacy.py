"""Per-learner export and erasure.

Right to access, right to be forgotten. Both are Phase 1 features, not a
later bolt-on.

The learner-scoped table list is derived from the SQLModel metadata: any
mapped table with a ``learner_id`` column is learner data. A table added
later is covered automatically, so new private data cannot be silently
missed by an export or left behind by a delete. Shared tables --
``objective``, ``resource``, ``item`` and the rest of the corpus -- have no
``learner_id`` and are therefore never touched.

The one special case is ``learner`` itself, whose own key is ``id``. It is
matched on that column instead. Nothing else is hard-coded.

Why this module deletes event rows directly
-------------------------------------------
:class:`the_oracle.store.events.EventLog` refuses ``update`` and ``delete``.
That rule is what makes every derived value reproducible: state is a pure
function of the log, so nothing may rewrite history. Erasure is the one
lawful exception. A learner who asks to be forgotten must have the raw
signal removed too, not just the derived tables, or the deletion is a lie.

The two rules are reconciled by scope, not by weakening the API:

* The append-only rule holds for all normal operation. No code path in the
  system may edit or remove an event. ``EventLog`` still raises.
* Erasure is not an operation *within* the model. It removes a learner from
  the system entirely. Afterwards there is no state to reproduce, so
  reproducibility is not violated -- it is vacuous.

So this module issues its deletes at the SQL layer, for whole learners
only, and never for a subset of one learner's events. Anything narrower
would be a history edit and is not allowed.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, Table, delete, select
from sqlmodel import SQLModel

from the_oracle.store import models as _models  # noqa: F401  (registers tables)
from the_oracle.store.db import get_engine, session_scope

LEARNER_COLUMN = "learner_id"
LEARNER_TABLE = "learner"
EVENT_TABLE = "event_log"


def _identity_column(table: Table) -> Any | None:
    """The column that carries the learner id, or ``None`` if shared."""
    if LEARNER_COLUMN in table.columns:
        return table.columns[LEARNER_COLUMN]
    if table.name == LEARNER_TABLE and "id" in table.columns:
        return table.columns["id"]
    return None


def _learner_scoped_tables() -> list[tuple[Table, Any]]:
    """Every learner-scoped table with its identity column, name-ordered."""
    pairs = [
        (table, column)
        for table in SQLModel.metadata.tables.values()
        if (column := _identity_column(table)) is not None
    ]
    return sorted(pairs, key=lambda pair: pair[0].name)


def learner_tables() -> list[str]:
    """Names of the learner-scoped tables, derived from the metadata."""
    return [table.name for table, _ in _learner_scoped_tables()]


def _jsonable(value: Any) -> Any:
    """Return a JSON-serialisable form of one column value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _order_by(table: Table) -> list[Any]:
    """Stable row order: primary key if there is one, else insertion order."""
    return list(table.primary_key.columns) or list(table.columns)[:1]


def export_learner(learner_id: str, engine: Engine | None = None) -> dict[str, Any]:
    """Everything the system holds about one learner, JSON-safe.

    Includes every learner-scoped table and the full event log. An unknown
    learner exports empty tables rather than raising.
    """
    engine = engine or get_engine()
    tables = _learner_scoped_tables()
    exported: dict[str, list[dict[str, Any]]] = {}

    with session_scope(engine) as session:
        for table, column in tables:
            statement = select(table).where(column == learner_id).order_by(*_order_by(table))
            rows = session.execute(statement).mappings().all()
            exported[table.name] = [
                {key: _jsonable(value) for key, value in dict(row).items()} for row in rows
            ]

    events = exported.get(EVENT_TABLE, [])
    return {
        "learner_id": learner_id,
        "exported_at": _models.utcnow().isoformat(),
        "tables": exported,
        "events": events,
        "counts": {name: len(rows) for name, rows in exported.items()},
    }


def delete_learner(learner_id: str, engine: Engine | None = None) -> dict[str, int]:
    """Erase one learner. Irreversible. Returns rows removed per table.

    Only rows whose ``learner_id`` matches are removed, so no other
    learner is affected and no shared table is touched. Deleting a learner
    who does not exist is a no-op that returns zero counts.
    """
    engine = engine or get_engine()
    tables = _learner_scoped_tables()
    counts: dict[str, int] = {}

    with session_scope(engine) as session:
        for table, column in tables:
            result = session.execute(delete(table).where(column == learner_id))
            counts[table.name] = int(result.rowcount or 0)

    return counts


__all__ = ["delete_learner", "export_learner", "learner_tables"]
