"""Split a monolithic skill-graph YAML into a shared library plus a manifest.

Usage:
    uv run python scripts/migrate_pack.py \
        data/skill_graphs/bayesian_forecasting.yaml data/packs

The source file is never modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

OBJECTIVE_KEYS = (
    "id",
    "version",
    "title",
    "description",
    "bloom",
    "difficulty",
    "est_minutes",
    "assessment_stems",
    "tags",
)


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def _tags(objective: dict[str, Any]) -> list[str]:
    """Derive coarse tags from the id prefix when the source has none."""
    tags = list(objective.get("tags") or [])
    if not tags:
        prefix = str(objective["id"]).split("_", 1)[0]
        tags = [prefix]
    return tags


def migrate(source: Path, out_root: Path, version: int = 1) -> tuple[int, Path]:
    """Write one YAML per objective plus one domain manifest. Returns (count, path)."""
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    objectives: list[dict[str, Any]] = data["objectives"]
    domain_id: str = data["domain"]

    for objective in objectives:
        body = {
            "id": objective["id"],
            "version": version,
            "title": objective["title"],
            "description": objective["description"],
            "bloom": objective["bloom"],
            "difficulty": objective["difficulty"],
            "est_minutes": objective["est_minutes"],
            "assessment_stems": list(objective.get("assessment_stems") or []),
            "tags": _tags(objective),
        }
        unknown = set(body) - set(OBJECTIVE_KEYS)
        if unknown:  # pragma: no cover - guard against schema drift
            msg = f"unexpected objective keys: {sorted(unknown)}"
            raise ValueError(msg)
        _dump(out_root / "objectives" / f"{body['id']}.yaml", body)

    edges = [
        {"from": prerequisite, "to": objective["id"]}
        for objective in objectives
        for prerequisite in objective.get("prerequisites") or []
    ]

    manifest = {
        "id": domain_id,
        "version": version,
        "title": data["title"],
        "description": " ".join(str(data.get("description", "")).split()),
        "objectives": [
            {"id": objective["id"], "version": version} for objective in objectives
        ],
        "edges": edges,
        "modules": [
            {
                "id": module["id"],
                "title": module["title"],
                "goal": " ".join(str(module["goal"]).split()),
                "objectives": list(module["objectives"]),
            }
            for module in data["modules"]
        ],
        "misconceptions": [
            {
                "id": mis["id"],
                "wrong_model": " ".join(str(mis["wrong_model"]).split()),
                "objectives": list(mis.get("objectives") or []),
                "diagnostic": " ".join(str(mis["diagnostic"]).split()),
            }
            for mis in data.get("misconceptions") or []
        ],
    }
    manifest_path = out_root / "domains" / f"{domain_id}.yaml"
    _dump(manifest_path, manifest)
    return len(objectives), manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="monolithic skill graph YAML")
    parser.add_argument("out_root", type=Path, help="pack root (objectives/, domains/)")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args(argv)

    count, manifest = migrate(args.source, args.out_root, args.version)
    print(f"wrote {count} objectives and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
