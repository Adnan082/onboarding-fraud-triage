"""FastAPI scoring service (CLAUDE.md section 10).

Run it with ``make serve``.

The artefacts load once, at startup, and their hashes are verified against the
manifest. If that fails the service refuses to start rather than serving scores
from a model nobody can identify -- a loud failure at deploy time is cheaper than
a quiet one in production.

``/health`` answers regardless, so a health check can distinguish "the process is
up but has no model" from "the process is down".
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from triage import __version__
from triage.api import service
from triage.api.schemas import HealthResponse, MonitorStatus, ScoreRequest, ScoreResponse

log = logging.getLogger("triage")

_state: dict[str, service.ScoringService | None] = {"service": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and verify the artefacts once, before the first request."""
    try:
        _state["service"] = service.ScoringService.load()
    except Exception as error:  # noqa: BLE001 - reported, then re-raised on use
        _state["service"] = None
        log.error("could not load scoring artefacts: %s", error)
    yield
    _state["service"] = None


app = FastAPI(
    title="onboarding-fraud-triage",
    version=__version__,
    summary="Scores an application and routes it to approve, review or extra verification.",
    lifespan=lifespan,
)


def _require_service() -> service.ScoringService:
    loaded = _state["service"]
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail="No scoring artefacts loaded. Run `make train` and `make conformal`.",
        )
    return loaded


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Answers whether or not a model is loaded."""
    return HealthResponse(status="ok")


@app.get("/version")
def version() -> dict:
    """Return ``models/manifest.json`` (hashes, config, git SHA)."""
    manifest = service.read_manifest()
    if manifest is None:
        raise HTTPException(status_code=503, detail="No model manifest yet: run `make train`.")
    return manifest


@app.get("/monitor/status", response_model=MonitorStatus)
def monitor_status() -> MonitorStatus:
    """Return what ``models/monitor_state.json`` currently says."""
    return MonitorStatus(**service.read_monitor_state())


@app.post("/score", response_model=ScoreResponse)
def score(
    request: ScoreRequest,
    explain: bool = Query(default=True, description="Set false to skip SHAP in latency tests"),
) -> ScoreResponse:
    """Score one application and return its decision band.

    Invalid input never reaches the model: the request body is generated from the
    frozen data contract, so an out-of-range value is a 422.
    """
    loaded = _require_service()
    features = request.features.model_dump()  # type: ignore[attr-defined]
    result = loaded.score(features, explain=explain)
    return ScoreResponse(application_id=request.application_id, **result)
