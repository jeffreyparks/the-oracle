"""The auth seam.

Phase 0 resolves a learner id from settings. When real auth arrives, only this
file changes: every caller already takes a :class:`LearnerContext`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from the_oracle.config import Settings, get_settings


class AuthError(RuntimeError):
    """Raised when no learner can be resolved."""


@dataclass(frozen=True, slots=True)
class LearnerContext:
    """Who is acting, and what they are allowed to spend.

    Treat this as opaque. Do not read ``learner_id`` off settings anywhere else.
    """

    learner_id: str
    settings: Settings
    claims: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def resolve(cls, settings: Settings | None = None) -> Self:
        """Resolve the acting learner.

        Phase 0: single local user from configuration. Later: a token, a
        session cookie, or a CLI keychain lookup. The signature does not move.
        """
        settings = settings or get_settings()
        learner_id = (settings.learner_id or "").strip()
        if not learner_id:
            raise AuthError("no learner id configured; set ORACLE_LEARNER_ID")
        return cls(learner_id=learner_id, settings=settings, claims={"source": "config"})

    @classmethod
    def for_learner(cls, learner_id: str, settings: Settings | None = None) -> Self:
        """Build a context for an explicit learner. Tests and admin paths only."""
        if not learner_id:
            raise AuthError("learner_id must not be empty")
        return cls(
            learner_id=learner_id,
            settings=settings or get_settings(),
            claims={"source": "explicit"},
        )

    @property
    def session_token_budget(self) -> int:
        return self.settings.session_token_budget

    @property
    def learner_token_budget(self) -> int:
        return self.settings.learner_token_budget

    def owns(self, learner_id: str) -> bool:
        """True when this context may read or write ``learner_id`` data."""
        return learner_id == self.learner_id

    def require(self, learner_id: str) -> None:
        """Raise unless this context owns ``learner_id``."""
        if not self.owns(learner_id):
            raise AuthError(f"context {self.learner_id!r} may not access {learner_id!r}")
