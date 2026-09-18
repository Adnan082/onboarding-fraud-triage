"""FastAPI scoring service (CLAUDE.md section 10).

Run it with ``make serve``. ``/score`` is wired up in week 3; ``/health``,
``/version`` and ``/monitor/status`` work as soon as their artefacts exist.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from triage import __version__
from triage.api import service
from triage.api.schemas import HealthResponse, MonitorStatus, ScoreRequest, ScoreResponse

app = FastAPI(
    title="onboarding-fraud-triage",
    version=__version__,
    summary="Scores an application and routes it to approve, review or extra verification.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
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
    """Score one application and return its decision band."""
    raise HTTPException(status_code=501, detail="Scoring is implemented in week 3.")
