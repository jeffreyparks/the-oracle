"""Regenerate objective tags deterministically from title and description.

Phase 1 debt item 3: the migrated objectives carry tags derived from their id
prefix, so dedupe recall is weak. This script rewrites them from the text that
actually describes the skill. No LLM, no cost, no network. Run it twice and the
files do not change on the second run.

Tags are recall aids, not semantic content, so ``version`` is never bumped.

Usage:
    uv run python scripts/backfill_tags.py data/packs/objectives
    uv run python scripts/backfill_tags.py data/packs/objectives --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

MAX_TAGS = 6
MIN_TOKEN_LEN = 3

#: A token must be in the title, or repeated, to become a tag. Without this a
#: run fills the last slots with whichever one-off description words sort first.
MIN_TAG_SCORE = 2.0

#: Title words count for more than description words: a title is the author's
#: own summary of the skill.
TITLE_WEIGHT = 3.0
DESCRIPTION_WEIGHT = 1.0

#: Generic English plus the vocabulary of instructional writing itself. These
#: words appear in most objectives, so they separate nothing.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been being between both but by can cannot could did do does
    doing done down during each either else every few for from further had has have
    having he her here hers him his how i if in into is it its itself just make makes
    many may me might more most much must my no nor not of off on once one only onto
    or other our out over own per same she should so some such than that the their
    them then there these they this those through to too under until up upon use used
    uses using very was way we were what when where whether which while who whom why
    will with within without would you your yours
    ability about above after again against all also am an any apply applies applied
    approach basic build builds check checks choose chooses common compare compares
    concept concepts course define defines describe describes different difference
    distinguish evaluate example examples explain explains fundamental general give
    gives good identify identifies interpret interprets introduction know knowledge
    learn learner learning level lesson material method methods module new objective
    objectives outcome practice practise problem problems produce read recognise
    recognize report result results say see set show shows simple skill skills state
    states student study students task tasks teach term terms test tests thing things
    topic understand understanding unit work write writes written
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9+-]*")

#: Order matters: longest suffix first, so "ations" is tried before "s".
_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ization", "ize"),
    ("izations", "ize"),
    ("ational", "ate"),
    ("ations", "ate"),
    ("ation", "ate"),
    ("ingly", ""),
    ("edly", ""),
    ("ings", ""),
    ("ing", ""),
    ("ies", "y"),
    ("ied", "y"),
    ("sses", "ss"),
    ("ses", "s"),
    ("es", ""),
    ("ed", ""),
    ("s", ""),
)


def stem(token: str) -> str:
    """Crude, deterministic suffix stripping. Good enough to group variants."""
    for suffix, replacement in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) + len(replacement) >= 4:
            return token[: -len(suffix)] + replacement
    return token


def tokenise(text: str) -> list[str]:
    """Lower-case word tokens, stopworded and length-filtered."""
    return [
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) >= MIN_TOKEN_LEN and token not in STOPWORDS
    ]


def tags_for(title: str, description: str, limit: int = MAX_TAGS) -> list[str]:
    """Deterministic keyword tags from title plus description.

    Tokens are stemmed to group variants, scored by weighted frequency, and
    rendered back as the shortest surface form seen. Ties break alphabetically,
    so the output never depends on dictionary order or on the run.
    """
    scores: Counter[str] = Counter()
    surface: dict[str, set[str]] = {}
    for text, weight in ((title, TITLE_WEIGHT), (description, DESCRIPTION_WEIGHT)):
        for token in tokenise(text):
            root = stem(token)
            if len(root) < MIN_TOKEN_LEN or root in STOPWORDS:
                continue
            scores[root] += weight
            surface.setdefault(root, set()).add(token)
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    strong = [pair for pair in ranked if pair[1] >= MIN_TAG_SCORE]
    return [sorted(surface[root], key=lambda s: (len(s), s))[0] for root, _ in strong[:limit]]


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def backfill(directory: Path, *, dry_run: bool = False) -> list[dict[str, Any]]:
    """Rewrite tags for every objective YAML in ``directory``. Returns a report."""
    report: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        body = _load(path)
        before = list(body.get("tags") or [])
        after = tags_for(str(body.get("title", "")), str(body.get("description", "")))
        changed = before != after
        if changed and not dry_run:
            body["tags"] = after
            _dump(path, body)
        report.append(
            {"id": body.get("id", path.stem), "before": before, "after": after,
             "changed": changed}
        )
    return report


def summarise(report: list[dict[str, Any]]) -> str:
    """A before/after summary a human can scan."""
    changed = [row for row in report if row["changed"]]
    before_total = sum(len(row["before"]) for row in report)
    after_total = sum(len(row["after"]) for row in report)
    vocab_before = {t for row in report for t in row["before"]}
    vocab_after = {t for row in report for t in row["after"]}
    lines = [
        f"{len(report)} objectives, {len(changed)} changed",
        f"tags: {before_total} -> {after_total} "
        f"(mean {before_total / max(len(report), 1):.2f} -> "
        f"{after_total / max(len(report), 1):.2f} per objective)",
        f"distinct tag vocabulary: {len(vocab_before)} -> {len(vocab_after)}",
        "",
    ]
    lines.extend(
        f"  {row['id']}: {row['before']} -> {row['after']}" for row in changed
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, nargs="?", default=Path("data/packs/objectives")
    )
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        print(f"no such directory: {args.directory}", file=sys.stderr)
        return 2
    report = backfill(args.directory, dry_run=args.dry_run)
    print(summarise(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
