"""Errors raised by the domains subpackage."""

from __future__ import annotations


class PackValidationError(Exception):
    """A domain pack failed validation. Carries every problem found."""

    def __init__(self, domain_id: str, problems: list[str]) -> None:
        self.domain_id = domain_id
        self.problems = list(problems)
        detail = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"domain {domain_id!r} failed validation:\n{detail}")


class ObjectiveNotFoundError(Exception):
    """An objective id (or pinned version) is absent from the shared library."""
