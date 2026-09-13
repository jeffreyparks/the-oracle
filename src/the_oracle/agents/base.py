"""The agent contract.

Built on Pydantic AI 2.43.0 (pinned; the 2.x line moves fast).

Every agent is a stateless object with a typed input, a typed output, usage
accounting, and an idempotency key. It never writes state. Only the
orchestrator writes state, and only the orchestrator checks the budget.

Phase 0 ships this base class and no real agents.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent
from sqlalchemy import Engine
from sqlmodel import Session, select

from the_oracle.config import Task, get_settings
from the_oracle.context import LearnerContext
from the_oracle.store import models
from the_oracle.store.db import get_engine, session_scope
from the_oracle.style import system_prompt

PYDANTIC_AI_VERSION = "2.43.0"

In = TypeVar("In", bound=BaseModel)
Out = TypeVar("Out", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    """Raised when a call would cross a token ceiling."""


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one agent call."""

    request_tokens: int = 0
    response_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.request_tokens + self.response_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            request_tokens=self.request_tokens + other.request_tokens,
            response_tokens=self.response_tokens + other.response_tokens,
        )

    @classmethod
    def from_run(cls, run_usage: Any) -> "Usage":
        """Read a Pydantic AI ``RunUsage`` without depending on its exact shape."""
        return cls(
            request_tokens=int(getattr(run_usage, "input_tokens", 0) or 0),
            response_tokens=int(getattr(run_usage, "output_tokens", 0) or 0),
        )


@dataclass(frozen=True, slots=True)
class IdempotencyKey:
    """``(agent, objective_id, input_hash)``. The retry guard."""

    agent: str
    objective_id: str
    input_hash: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.agent, self.objective_id, self.input_hash)


@dataclass(frozen=True, slots=True)
class AgentResult(Generic[Out]):
    """What an agent returns: the typed output, the cost, and the cache flag."""

    output: Out
    usage: Usage = field(default_factory=Usage)
    key: IdempotencyKey | None = None
    model: str | None = None
    cached: bool = False
    created_at: datetime = field(default_factory=models.utcnow)


def hash_input(payload: BaseModel | dict[str, Any]) -> str:
    """Stable SHA-256 over a canonical JSON form of the agent input."""
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Agent(ABC, Generic[In, Out]):
    """Typed agent base.

    Subclasses set :attr:`name`, :attr:`task`, :attr:`input_type`,
    :attr:`output_type`, and :attr:`role`, then implement :meth:`_run`.
    The default :meth:`_run` calls the routed model through Pydantic AI.
    """

    name: ClassVar[str]
    task: ClassVar[Task]
    role: ClassVar[str] = "You are one specialist inside a tutoring system."
    input_type: ClassVar[type[BaseModel]]
    output_type: ClassVar[type[BaseModel]]

    def __init__(self, engine: Engine | None = None, *, model: str | None = None) -> None:
        settings = get_settings()
        self._engine = engine or get_engine(settings)
        self._model = model or settings.model_for(self.task)
        self._agent: PydanticAgent[None, Any] | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def engine(self) -> Engine:
        return self._engine

    def pydantic_agent(self) -> PydanticAgent[None, Any]:
        """Lazily build the underlying Pydantic AI agent. Stateless, reusable."""
        if self._agent is None:
            self._agent = PydanticAgent(
                self._model,
                output_type=self.output_type,
                instructions=system_prompt(self.role),
                name=self.name,
            )
        return self._agent

    # -- idempotency -------------------------------------------------------

    def key_for(self, payload: In, objective_id: str = "") -> IdempotencyKey:
        return IdempotencyKey(
            agent=self.name,
            objective_id=objective_id or "",
            input_hash=hash_input(payload),
        )

    def lookup(self, key: IdempotencyKey) -> AgentResult[Out] | None:
        """Return the cached result for ``key``, or ``None``."""
        statement = (
            select(models.AgentCall)
            .where(models.AgentCall.agent == key.agent)
            .where(models.AgentCall.objective_id == key.objective_id)
            .where(models.AgentCall.input_hash == key.input_hash)
        )
        with Session(self._engine) as session:
            row = session.exec(statement).first()
            if row is None:
                return None
            output = self.output_type.model_validate(row.output)
            return AgentResult(
                output=output,  # type: ignore[arg-type]
                usage=Usage(row.request_tokens, row.response_tokens),
                key=key,
                model=row.model,
                cached=True,
                created_at=row.created_at,
            )

    def record(self, key: IdempotencyKey, result: AgentResult[Out]) -> None:
        """Write one call into the idempotency cache. Last writer wins is fine."""
        with session_scope(self._engine) as session:
            existing = session.exec(
                select(models.AgentCall)
                .where(models.AgentCall.agent == key.agent)
                .where(models.AgentCall.objective_id == key.objective_id)
                .where(models.AgentCall.input_hash == key.input_hash)
            ).first()
            if existing is not None:
                return
            session.add(
                models.AgentCall(
                    agent=key.agent,
                    objective_id=key.objective_id,
                    input_hash=key.input_hash,
                    model=result.model or self._model,
                    output=result.output.model_dump(mode="json"),
                    request_tokens=result.usage.request_tokens,
                    response_tokens=result.usage.response_tokens,
                    total_tokens=result.usage.total_tokens,
                )
            )

    # -- execution ---------------------------------------------------------

    async def run(
        self,
        ctx: LearnerContext,
        payload: In,
        *,
        objective_id: str = "",
        force: bool = False,
    ) -> AgentResult[Out]:
        """Run the agent, returning the cached result on a retry.

        The budget is *not* checked here. The orchestrator is the single cost
        chokepoint; see PLAN.md section 9.
        """
        key = self.key_for(payload, objective_id)
        if not force:
            cached = self.lookup(key)
            if cached is not None:
                return cached

        output, usage = await self._run(ctx, payload)
        result: AgentResult[Out] = AgentResult(
            output=output, usage=usage, key=key, model=self._model, cached=False
        )
        self.record(key, result)
        return result

    @abstractmethod
    async def _run(self, ctx: LearnerContext, payload: In) -> tuple[Out, Usage]:
        """Do the real work. Subclasses implement this and nothing else."""

    async def _run_llm(self, prompt: str) -> tuple[Out, Usage]:
        """Helper: one Pydantic AI call, typed output, usage extracted."""
        run = await self.pydantic_agent().run(prompt)
        return run.output, Usage.from_run(run.usage())
