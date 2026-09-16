"""Engine settings. No subject matter here, ever."""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_dotenv_files() -> list[Path]:
    """Load provider credentials from .env into the real process environment.

    pydantic-settings only maps ``ORACLE_``-prefixed variables onto Settings
    fields. Model SDKs read their own keys straight from ``os.environ``
    (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``SERPER_API_KEY``, ...), so a
    key sitting in a .env file would otherwise be invisible to them.

    Search order, first file wins per key, and a real environment variable
    always beats a file:

    1. ``$ORACLE_HOME/.env`` (or the default home), so a deployed install keeps
       its secrets beside its data.
    2. ``.env`` in the current directory and each parent, so a checkout works
       from any subdirectory.
    """
    loaded: list[Path] = []
    candidates: list[Path] = []

    home_raw = os.environ.get("ORACLE_HOME")
    home = Path(home_raw).expanduser() if home_raw else Path.home() / ".the-oracle"
    candidates.append(home / ".env")

    here = Path.cwd().resolve()
    candidates.extend(parent / ".env" for parent in (here, *here.parents))

    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded


#: Populated at import time. ``oracle version`` reports it.
DOTENV_FILES: list[Path] = _load_dotenv_files()


class Task(StrEnum):
    """Routing keys. One per LLM-using capability, plus cheap utility work."""

    ASSESS = "assess"
    ARCHITECT = "architect"
    SCOUT = "scout"
    CRITIC = "critic"
    AUTHOR = "author"
    COACH = "coach"
    SUMMARISE = "summarise"
    EMBED = "embed"


DEFAULT_MODEL_ROUTES: dict[Task, str] = {
    # Strong models where judgment is the product.
    Task.ASSESS: "anthropic:claude-sonnet-4-5",
    Task.ARCHITECT: "anthropic:claude-sonnet-4-5",
    Task.AUTHOR: "anthropic:claude-sonnet-4-5",
    Task.CRITIC: "anthropic:claude-sonnet-4-5",
    # Cheap models for fetch, skim, and summarise.
    Task.SCOUT: "anthropic:claude-haiku-4-5",
    Task.COACH: "anthropic:claude-haiku-4-5",
    Task.SUMMARISE: "anthropic:claude-haiku-4-5",
    Task.EMBED: "openai:text-embedding-3-small",
}


class Settings(BaseSettings):
    """Process-wide settings. Environment prefix ``ORACLE_``."""

    model_config = SettingsConfigDict(
        env_prefix="ORACLE_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    home: Path = Field(default=Path.home() / ".the-oracle")
    """Root of the user data directory. ``ORACLE_HOME`` overrides it."""

    database_url: str | None = None
    """Full SQLAlchemy URL. Default is SQLite inside :attr:`home`."""

    learner_id: str = "default"
    """Phase 0 auth stub. See :mod:`the_oracle.context`."""

    model_routes: dict[Task, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_ROUTES))

    session_token_budget: int = 120_000
    """Ceiling for one study session. Checked by the orchestrator only."""

    learner_token_budget: int = 5_000_000
    """Lifetime ceiling per learner."""

    telemetry: bool = True
    """Turn Logfire instrumentation on. Nothing leaves the machine without
    ``LOGFIRE_TOKEN``. ``ORACLE_TELEMETRY=0`` disables it outright."""

    telemetry_service_name: str = "the-oracle"
    telemetry_environment: str = ""
    """Free text, e.g. ``dev`` or ``prod``. Empty means unset."""

    telemetry_capture_content: bool = False
    """Send prompts and completions to Logfire. Off: learner text is private."""

    default_session_minutes: int = 25
    max_diagnostic_items: int = 12
    mastery_threshold: float = 0.85

    @field_validator("home", mode="after")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser()

    @property
    def objectives_dir(self) -> Path:
        return self.home / "objectives"

    @property
    def domains_dir(self) -> Path:
        return self.home / "domains"

    @property
    def db_path(self) -> Path:
        return self.home / "oracle.db"

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.db_path}"

    def model_for(self, task: Task) -> str:
        """Return the model selector routed to ``task``."""
        try:
            return self.model_routes[task]
        except KeyError as exc:  # pragma: no cover - guarded by the enum
            raise KeyError(f"no model route for task {task!r}") from exc

    def ensure_home(self) -> Path:
        """Create the user data directory tree if it is missing."""
        self.objectives_dir.mkdir(parents=True, exist_ok=True)
        self.domains_dir.mkdir(parents=True, exist_ok=True)
        return self.home


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. Tests use this after changing the environment."""
    get_settings.cache_clear()
