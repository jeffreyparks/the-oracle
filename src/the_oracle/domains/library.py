"""The shared objective library: one YAML file per objective."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from the_oracle.domains.errors import ObjectiveNotFoundError
from the_oracle.domains.schema import Objective

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator
    from pathlib import Path


def _library_dir() -> Path:
    from the_oracle.domains.registry import oracle_home

    return oracle_home() / "objectives"


def _read(path: Path) -> Objective:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Objective.model_validate(data)


def get(objective_id: str, version: int | None = None) -> Objective:
    """Load one objective. Raises ObjectiveNotFoundError on miss or version skew."""
    path = _library_dir() / f"{objective_id}.yaml"
    if not path.is_file():
        msg = f"objective {objective_id!r} not in library at {path}"
        raise ObjectiveNotFoundError(msg)
    obj = _read(path)
    if obj.id != objective_id:
        msg = f"objective file {path} declares id {obj.id!r}"
        raise ObjectiveNotFoundError(msg)
    if version is not None and obj.version != version:
        msg = (
            f"objective {objective_id!r} is version {obj.version}, "
            f"pinned version {version} requested"
        )
        raise ObjectiveNotFoundError(msg)
    return obj


def put(obj: Objective) -> None:
    """Write an objective into the library, creating the directory if needed."""
    directory = _library_dir()
    directory.mkdir(parents=True, exist_ok=True)
    payload = obj.model_dump(mode="json")
    (directory / f"{obj.id}.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def all_objectives() -> Iterator[Objective]:
    """Yield every objective in the library, ordered by id."""
    directory = _library_dir()
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.yaml")):
        yield _read(path)


def search(text: str, limit: int = 20) -> list[Objective]:
    """Cheap substring search over id, title, description, and tags."""
    needle = text.casefold().strip()
    hits: list[Objective] = []
    for obj in all_objectives():
        haystack = " ".join(
            [obj.id, obj.title, obj.description, " ".join(obj.tags)]
        ).casefold()
        if needle in haystack:
            hits.append(obj)
        if len(hits) >= limit:
            break
    return hits
