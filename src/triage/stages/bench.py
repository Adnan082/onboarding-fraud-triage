"""Latency benchmark -> reports/metrics.json:service.

Section 8.3: p50 and p99 over 2,000 sequential requests after 100 warm-up, on CPU
only. Two numbers are reported, because they answer different questions:

- **scoring** is the model path alone, which is what a batch job would see;
- **HTTP** adds serialisation, validation against the contract, and the framework,
  which is what a caller actually waits for.

Both with and without SHAP, because reason codes dominate the cost and a service
that does not need them should not pay for them.

Whatever number comes out is reported. The target in section 10 is "p50 under
about 15 ms", and if it is missed that is a finding, not something to tune away
quietly.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from fastapi.testclient import TestClient
from omegaconf import DictConfig

from triage.data.contract import CONTRACT
from triage.evaluation.artefacts import update_section
from triage.stages._base import CONFIG_PATH, stage_run

log = logging.getLogger("triage")

WARMUP = 100
REQUESTS = 2_000
DOCKER_IMAGE = "onboarding-fraud-triage:dev"


def example_application() -> dict[str, Any]:
    """One legal application, generated from the contract."""
    from triage.api.schemas import EXCLUDED_FROM_REQUEST

    payload: dict[str, Any] = {}
    for name, spec in CONTRACT.items():
        if name in EXCLUDED_FROM_REQUEST:
            continue
        if spec.levels:
            payload[name] = spec.levels[0]
        else:
            low = float(spec.lo if spec.lo is not None else 0.0)
            high = float(spec.hi if spec.hi is not None else 1.0)
            middle = low + (high - low) / 2
            payload[name] = int(middle) if spec.kind == "int" else middle
    return payload


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    """The numbers section 8.3 asks for, plus enough context to read them."""
    values = np.asarray(samples_ms, dtype=float)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "n": float(values.size),
    }


def time_calls(call, n: int, warmup: int) -> list[float]:
    """Time ``n`` sequential calls in milliseconds, after discarding ``warmup``."""
    for _ in range(warmup):
        call()

    samples = []
    for _ in range(n):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def docker_image_size() -> dict[str, Any]:
    """The built image's size, if it has been built. Never builds it here."""
    if shutil.which("docker") is None:
        return {"available": False, "reason": "docker is not on PATH"}

    try:
        result = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE, "--format", "{{.Size}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"available": False, "reason": f"{DOCKER_IMAGE} is not built: run `make docker`"}

    size_bytes = int(result.stdout.strip())
    return {
        "available": True,
        "image": DOCKER_IMAGE,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1e6, 1),
    }


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``make bench``."""
    with stage_run(cfg, "bench") as context:
        from triage.api.app import app
        from triage.api.service import ScoringService

        features = example_application()
        # Pass the stage's own config: Hydra is already initialised here, so the
        # service must not compose a second one.
        service = ScoringService.load(cfg)
        log.info("benchmarking %s on CPU", service.manifest["model_version"])

        results: dict[str, Any] = {}

        for explain in (True, False):
            label = "with_shap" if explain else "without_shap"
            # Bind the loop variable: a late-binding closure here would silently
            # benchmark the wrong configuration if this were ever deferred.
            samples = time_calls(
                lambda wanted=explain: service.score(features, explain=wanted), REQUESTS, WARMUP
            )
            results[f"scoring_{label}"] = percentiles(samples)
            log.info(
                "  scoring %-13s p50 %.2f ms, p99 %.2f ms",
                label,
                results[f"scoring_{label}"]["p50_ms"],
                results[f"scoring_{label}"]["p99_ms"],
            )

        with TestClient(app) as client:
            for explain in (True, False):
                label = "with_shap" if explain else "without_shap"
                url = f"/score?explain={'true' if explain else 'false'}"
                body = {"features": features}

                # Never benchmark an error path: a 503 is fast and means nothing.
                probe = client.post(url, json=body)
                if probe.status_code != 200:
                    raise RuntimeError(
                        f"the service returned {probe.status_code} for {url}: "
                        f"{probe.text[:200]}. Refusing to report latency for a failing request."
                    )

                samples = time_calls(
                    lambda where=url, payload=body: client.post(where, json=payload),
                    REQUESTS,
                    WARMUP,
                )
                results[f"http_{label}"] = percentiles(samples)
                log.info(
                    "  http    %-13s p50 %.2f ms, p99 %.2f ms",
                    label,
                    results[f"http_{label}"]["p50_ms"],
                    results[f"http_{label}"]["p99_ms"],
                )

        image = docker_image_size()
        if image["available"]:
            log.info("  docker image %s: %.1f MB", image["image"], image["size_mb"])
        else:
            log.info("  docker image: %s", image["reason"])

        payload = {
            "model_version": service.manifest["model_version"],
            "requests": REQUESTS,
            "warmup": WARMUP,
            "device": "cpu",
            "target_p50_ms": 15.0,
            "latency": results,
            "docker": image,
            "note": (
                "Sequential, single process, in-process HTTP client (no network). "
                "Reason codes dominate: compare with_shap against without_shap."
            ),
        }
        update_section(Path(cfg.paths.metrics), "service", payload, context)
        log.info("wrote %s -> service", cfg.paths.metrics)


if __name__ == "__main__":
    main()
