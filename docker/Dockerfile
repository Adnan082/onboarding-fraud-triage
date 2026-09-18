# Service image: python:3.11-slim, uv, non-root user, healthcheck.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so the layer caches. Linux image: the fairgbm extra is installed.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra fairgbm

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev --extra fairgbm

# Model artefacts are mounted at run time; they are not baked into the image.
RUN useradd --create-home --uid 10001 triage \
 && mkdir -p /app/models /app/reports \
 && chown -R triage:triage /app
USER triage

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "triage.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
