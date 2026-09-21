# onboarding-fraud-triage
# POSIX shell assumed (WSL2 / Linux / macOS / the Docker image). See CLAUDE.md section 5.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV      ?= uv
RUN     := $(UV) run
PY      := $(RUN) python
STAGE   := $(PY) -m triage.stages
DATASET := sgpjesus/bank-account-fraud-dataset-neurips-2022
# Only the three variants this project uses: the other three are ~680 MB we never read.
VARIANTS := "Base.csv" "Variant IV.csv" "Variant V.csv"
PORT    ?= 8000
IMAGE   ?= onboarding-fraud-triage:dev

# Extra Hydra overrides, e.g. `make train ARGS="data.sample_frac=0.1"`.
ARGS ?=

.PHONY: help setup data contract baseline train conformal fairness monitor evaluate \
        report serve bench demo test test-data lint format docker all clean

help:  ## List the targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## uv sync with the dev group (plus the fairgbm extra on Linux); install pre-commit hooks
	@if [ "$$(uname -s)" = "Linux" ]; then $(UV) sync --extra fairgbm; else $(UV) sync; fi
	$(RUN) pre-commit install
	$(RUN) nbstripout --install --attributes .gitattributes || true

data:  ## Download BAF, verify checksums, write data/interim/*.parquet
	@test -f "$$HOME/.kaggle/kaggle.json" || [ -n "$${KAGGLE_USERNAME:-}" ] || \
		{ echo "Need ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY (see CLAUDE.md section 7)."; exit 1; }
	@for f in $(VARIANTS); do \
		echo "--- $$f"; \
		$(RUN) kaggle datasets download $(DATASET) -f "$$f" -p data/raw --unzip; \
	done
	# `--unzip` is ignored when `-f` is given: files land as *.zip with URL-encoded
	# names, so extract here. Each archive carries the correct name inside.
	$(PY) -c "import zipfile, pathlib; [zipfile.ZipFile(z).extractall('data/raw') for z in pathlib.Path('data/raw').glob('*.zip')]"
	$(PY) -m triage.data.load $(ARGS)

contract:  ## Validate data/interim against the pandera contract; write reports/data_contract.md
	$(PY) -m triage.data.contract $(ARGS)

baseline:  ## B0 and B1 under both protocols -> reports/metrics.json:baselines
	$(STAGE).baseline $(ARGS)

train:  ## Champion + probability calibration -> models/, models/manifest.json
	$(STAGE).train $(ARGS)

conformal:  ## Alpha sweep on cal_tune, thresholds on cal_conf -> reports/tables/policy_grid.csv
	$(STAGE).conformal $(ARGS)

fairness:  ## Mitigation experiments M1-M4, bootstrap CIs, trade-off plot
	$(STAGE).fairness $(ARGS)

monitor:  ## Clean windows, variant stress, injected bugs -> reports/monitoring.json
	$(STAGE).monitor $(ARGS)

evaluate: baseline train conformal fairness monitor  ## Run the whole evaluation chain

report:  ## Regenerate README tables and reports/figures/ from artefacts
	$(STAGE).report $(ARGS)

serve:  ## Run the scoring API
	$(RUN) uvicorn triage.api.app:app --port $(PORT)

bench:  ## Latency benchmark -> reports/metrics.json:service
	$(STAGE).bench $(ARGS)

demo:  ## One-screen Streamlit demo (reads precomputed artefacts only)
	$(RUN) streamlit run app/demo.py

test:  ## Fast, data-free tests
	$(RUN) pytest -m "not data"

test-data:  ## Tests that need data/interim/
	$(RUN) pytest -m data

# Section 13 sets the bar at 85% for these four packages only. They are where a
# silent mistake would be worst: the guarantee, the fairness numbers, the
# detectors and the decision rule. The rest of the tree is measured, not gated.
coverage:  ## Enforce the section 13 coverage bar on the load-bearing packages
	$(RUN) pytest -m "not data" \
	  --cov=src/triage/uncertainty --cov=src/triage/fairness \
	  --cov=src/triage/monitoring --cov=src/triage/policy \
	  --cov-report=term-missing --cov-fail-under=85

lint:  ## ruff check, ruff format --check, mypy
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy src

format:  ## Apply ruff fixes and formatting
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

docker:  ## Build the service image (includes the fairgbm extra)
	docker build -f docker/Dockerfile -t $(IMAGE) .

all: data contract evaluate report bench test  ## Reproduce every artefact from a clean clone

clean:  ## Remove caches and generated figures (never touches data/ or reports/*.json)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
