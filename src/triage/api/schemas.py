"""Request and response models for the scoring API (CLAUDE.md section 10).

``ApplicationFeatures`` is generated from the frozen data contract with
``extra="forbid"``, so it enforces ranges and category sets and returns 422 on
anything invalid -- including the out-of-contract ``income x 10`` bug.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["approve", "review", "verify"]
ClassLabel = Literal["legit", "fraud"]


class Reason(BaseModel):
    """One analyst-facing reason code. Never shown to an applicant."""

    feature: str
    direction: Literal["high", "low"]
    text: str


class ScoreRequest(BaseModel):
    """One application to score."""

    model_config = ConfigDict(extra="forbid")

    application_id: str | None = None
    # TODO(week 3): replace with the contract-generated model (extra="forbid",
    # ranges and category sets enforced). customer_age is accepted and logged for
    # fairness monitoring, then dropped before scoring.
    features: dict[str, object] = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    """The scoring result and the policy that produced it."""

    application_id: str | None = None
    risk_score: float
    decision: Decision
    conformal_set: list[ClassLabel]
    reasons: list[Reason] = Field(default_factory=list)
    model_version: str
    policy_version: str
    drift_status: str
    fallback_active: bool


class HealthResponse(BaseModel):
    """Liveness only."""

    status: Literal["ok"]


class MonitorStatus(BaseModel):
    """What ``models/monitor_state.json`` currently says."""

    drift_status: str
    fallback_active: bool
    last_window: int | None = None
    updated_at: str | None = None
