"""Request and response models for the scoring API (CLAUDE.md section 10).

``ApplicationFeatures`` is **generated from the frozen data contract** rather than
written out by hand, so the API and the batch pipeline cannot drift apart. It sets
``extra="forbid"`` and enforces the contract's ranges and category sets, which is
what makes the ``income x 10`` bug a 422 at the door rather than a bad score.

The label and the month are deliberately absent: a live application has no outcome
and no month index. ``customer_age`` is present because it is logged for fairness
monitoring, and dropped before scoring (rule 4).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from triage.data.contract import CONTRACT, ColumnSpec

Decision = Literal["approve", "review", "verify"]
ClassLabel = Literal["legit", "fraud"]

# Never part of a scoring request: the outcome, and the split key.
EXCLUDED_FROM_REQUEST = ("fraud_bool", "month")


def _field_for(spec: ColumnSpec) -> tuple[Any, Any]:
    """One pydantic field from one contract column."""
    description = spec.note or None

    if spec.levels:
        if spec.kind == "category":
            annotation: Any = Literal[tuple(str(level) for level in spec.levels)]
        else:
            annotation = Literal[tuple(int(level) for level in spec.levels)]
        return annotation, Field(..., description=description)

    base = int if spec.kind == "int" else float
    return (
        Annotated[base, Field(ge=spec.lo, le=spec.hi)],
        Field(..., description=description),
    )


def build_features_model() -> type[BaseModel]:
    """The request body's ``features``, generated from the contract."""
    fields = {
        name: _field_for(spec)
        for name, spec in CONTRACT.items()
        if name not in EXCLUDED_FROM_REQUEST
    }
    return create_model(
        "ApplicationFeatures",
        __config__=ConfigDict(extra="forbid"),
        **fields,  # type: ignore[call-overload]
    )


ApplicationFeatures = build_features_model()


class Reason(BaseModel):
    """One analyst-facing reason code. Never shown to an applicant."""

    feature: str
    direction: Literal["high", "low"]
    text: str


class ScoreRequest(BaseModel):
    """One application to score."""

    model_config = ConfigDict(extra="forbid")

    application_id: str | None = None
    features: ApplicationFeatures  # type: ignore[valid-type]


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

    model_config = ConfigDict(extra="allow")

    drift_status: str
    fallback_active: bool
    last_window: int | None = None
    updated_at: str | None = None
