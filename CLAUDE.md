# CLAUDE.md: onboarding-fraud-triage

Instructions for Claude Code. Read this whole file at the start of every session.
Session notes go in `docs/PROGRESS.md` and lasting design decisions in `docs/DECISIONS.md`.

---

## 1. The project

A portfolio project. It scores online bank-account applications for fraud and routes each one to one of three outcomes: **approve**, **human review** or **extra verification**. It uses the public Bank Account Fraud (BAF) dataset from NeurIPS 2022.

Four things make it more than a Kaggle notebook:

1. **A calibrated decision policy.** Label-conditional conformal prediction guarantees two things. A stated share of fraud is never auto-approved, and at most a stated share of genuine applicants is sent to extra verification. Both are sized to what a review team can handle.
2. **A fairness check.** False alarms are compared by age with confidence intervals, and mitigation experiments are run.
3. **A label-free monitor.** It detects shifted populations and upstream data bugs, and falls back to more human review.
4. **Shippable outputs.** A tested, containerised FastAPI service, a one-screen demo, and a two-page validation report in the style of a UK bank model validation (PRA SS1/23).

**Owner:** Adnan (MSc Applied Data Science). The build is about 15 evenings of roughly 2.5 hours each over three weeks, for UK bank graduate applications.

**Audiences:**
- a non-technical assessor, who needs a 60-second demo;
- a model-risk reviewer, who reads the validation report;
- an engineer, who reads the code, tests and API.

**Non-goals:**
- real customer data;
- production deployment;
- LLM features before v1.0;
- any claim about a named bank. Don't name a bank anywhere in the repo.

---

## 2. Hard rules

1. **Split by time only.** Never let months overlap between training and evaluation. No random row splits, and no cross-validation that shuffles months.
2. **Never tune anything on months 6–7.** The one exception is the "paper protocol" threshold, which the BAF paper sets on test data. Label it as such wherever it appears.
3. **Keep row-level data out of git.** Never commit `data/`, row-level `*.csv`/`*.parquet`, `mlruns/`, `models/*.joblib`, `kaggle.json` or `.env`.
   - Aggregate artefacts in `reports/` *are* committed.
   - Tiny synthetic fixtures under `tests/fixtures/` are fine.
4. **Use age to measure, never to decide.** The champion model excludes `customer_age` (`features.use_age: false`).
   - Age may be used to compute fairness metrics.
   - Age may be used as a training-time constraint group in the FairGBM/fairlearn experiments.
   - Never use age, or age-specific thresholds, at decision time. Any comparison that does must be labelled *"analysis only: needs legal sign-off"*.
5. **Never decline automatically.** The riskiest band is "verify" (step-up checks or a human).
6. **Take every reported number from a generated artefact.** README tables, the report and the demo read only from `reports/metrics.json` and `reports/tables/`. Never type, estimate or recall a result. If an artefact is missing, run the make target or say it is missing.
7. **Write conformal prediction from scratch** in `src/triage/uncertainty/conformal.py`. MAPIE is only for cross-checking, in tests or notebooks.
8. **Report failures.** A method that doesn't work goes into "What didn't work" and `docs/PROGRESS.md`. Never delete a failed experiment to tidy the story.
9. **Scope every claim.**
   - BAF is synthetic, generated from an unnamed bank's applications. It is not UK data.
   - Coverage guarantees hold only while new data is exchangeable with the calibration data.
   - Costs are illustrative parameters. Never claim "£ saved".
   - Flag legal questions; never answer them.
10. **Make it reproducible from a clean clone.** Seed everything. `make all` must regenerate every artefact.
11. **Ask first** before adding a dependency, changing a metric definition, changing split boundaries, or touching anything in §3.

---

## 3. Decisions reserved for the owner

Ask the owner; don't assume. Defaults in `configs/policy/default.yaml` are placeholders tagged `# ASSUMPTION`.

- Review capacity (the share of applications the team can review) and all cost parameters.
- The α targets and the fallback α preset.
- Whether `proposed_credit_limit` is an acceptable proxy for the loss on a missed fraud.
- Fairness mitigation choices, and anything that involves legal interpretation.
- Anything that publishes data. The owner checks the licence on the Kaggle page first.

---

## 4. Environment and dependencies

- **Python and tooling.** Use Python 3.11 with `uv`. Develop in WSL2 (Ubuntu) or the Docker image; the Makefile assumes a POSIX shell.
- **Pinned stack.** These versions were checked to resolve together on Python 3.11 in September 2026:
  - `scikit-learn==1.5.2`
  - `pandas>=2.2,<3` (resolved to 2.3.3), `numpy` 2.x, `pyarrow`
  - `lightgbm` 4.7, `fairlearn` 0.14, `mapie` 1.5, `shap` 0.51
  - `pandera` 0.33, `duckdb` 1.5, `hydra-core==1.3.2`, `mlflow` 3.x
  - `fastapi`, `uvicorn[standard]`, `httpx`, `pydantic` 2, `streamlit`
  - `matplotlib`, `joblib`, `kaggle` 2.x
  - Dev group: `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`, `pre-commit`, `nbstripout`.
- **Why scikit-learn is pinned to 1.5.2.** `fairgbm==0.9.14` (last released Nov 2022) passes a scikit-learn argument that newer versions removed. It failed on scikit-learn 1.9.1 and trained fine on 1.5.2. Don't upgrade scikit-learn without re-running `tests/test_fairgbm_smoke.py`.
- **FairGBM runs on Linux only**, because it bundles a Linux `.so`. Set it up like this:
  - Declare it as an optional extra with a platform marker: `fairgbm = ["fairgbm==0.9.14; sys_platform == 'linux'"]`.
  - Import it lazily, and skip its experiment with a clear message when it isn't available.
  - Docker and CI install the extra.
- **fairlearn** (`ExponentiatedGradient` + `FalsePositiveRateParity`) works on every platform, and `predict(X)` needs no sensitive feature. It returns a randomised classifier at a single operating point, so evaluate it at that point rather than with threshold-sweep metrics.
- **pandera.** Import it as `import pandera.pandas as pa`.
- **MAPIE 1.x API.**
  - Construct `SplitConformalClassifier(estimator=..., confidence_level=0.9, conformity_score="lac", prefit=True)`.
  - Call `.conformalize(X, y)`, then `.predict_set(X)`. It returns `(predictions, sets)`, with `sets` shaped `(n, n_classes, n_levels)`.
  - It is *marginal*, not label-conditional, and at ~1% prevalence it produces many empty sets. Use it to check a marginal version of your code, not the label-conditional policy.
- **Kaggle CLI 2.x** takes the dataset as a positional argument, not `-d`. See §7.
- **MLflow.** Use a local file store in `mlruns/` (git-ignored). For each stage, log the params, metrics, config hash, data checksums and git SHA.

---

## 5. Commands

| Command | What it does |
|---|---|
| `make setup` | `uv sync` with the dev group (adds the `fairgbm` extra on Linux); `pre-commit install` |
| `make data` | Downloads BAF (needs `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`), verifies `data/checksums.sha256`, writes `data/interim/{base,variant_iv,variant_v}.parquet` |
| `make contract` | Validates interim data against the pandera contract and writes `reports/data_contract.md` |
| `make baseline` | Runs B0 and B1 under both protocols and writes `reports/metrics.json` → `baselines` |
| `make train` | Trains the champion model plus probability calibration. Writes `models/` and `models/manifest.json` (hashes, config, git SHA) |
| `make conformal` | Runs the α sweep on `cal_tune` and computes thresholds on `cal_conf`. Writes test-month results and `reports/tables/policy_grid.csv` |
| `make fairness` | Runs mitigation experiments M1–M4, bootstrap CIs and the trade-off plot |
| `make monitor` | Runs clean windows, variant stress tests and injected bugs. Writes `reports/monitoring.json` and `models/monitor_state.json` |
| `make evaluate` | Runs `baseline train conformal fairness monitor` (each target is added as it is built) |
| `make report` | Regenerates README tables (between `<!-- metrics:NAME -->` markers) and `reports/figures/` from artefacts |
| `make serve` | `uvicorn triage.api.app:app --port 8000` |
| `make bench` | Runs the latency benchmark and writes `reports/metrics.json` → `service` |
| `make demo` | `streamlit run app/demo.py` |
| `make test` | Fast, data-free tests: `pytest -m "not data"` |
| `make test-data` | Tests that need `data/interim/`: `pytest -m data` |
| `make lint` | `ruff check`, `ruff format --check`, `mypy src` |
| `make docker` | Builds the service image, including the `fairgbm` extra |
| `make all` | Runs `data contract evaluate report bench test` |

Stages are Hydra apps under `src/triage/stages/`. Run one with `uv run python -m triage.stages.<name> [overrides]`.

While developing, use `data.sample_frac=0.1`. Never report results from a sampled run.

---

## 6. Repo layout

```text
onboarding-fraud-triage/
├── CLAUDE.md  README.md  Makefile  pyproject.toml  uv.lock
├── .pre-commit-config.yaml  .gitignore  .github/workflows/ci.yml
├── configs/
│   ├── config.yaml               # defaults list, seed, paths
│   ├── data/baf.yaml             # files, months, window_size: 4000, sample_frac
│   ├── features/default.yaml     # use_age: false, categoricals, sentinels
│   ├── model/{logreg,lgbm,champion,fairgbm,fairlearn_eg}.yaml
│   ├── calibration/default.yaml  # none | platt | isotonic; select_by: brier
│   ├── policy/default.yaml       # alphas, capacity, costs (# ASSUMPTION), fallback preset
│   ├── monitor/default.yaml      # detectors, thresholds, bug specs
│   └── reasons.yaml              # feature + direction -> plain-English analyst text
├── data/                         # git-ignored except README.md and checksums.sha256
│   ├── raw/
│   └── interim/
├── src/triage/
│   ├── data/          load.py · contract.py · split.py
│   ├── features/      encode.py (fit on train only) · sentinels.py
│   ├── models/        baselines.py · champion.py · calibrate.py · fair.py
│   ├── uncertainty/   conformal.py
│   ├── policy/        decide.py · cost.py · sweep.py
│   ├── fairness/      metrics.py · bootstrap.py
│   ├── monitoring/    psi.sql · psi.py · domain_clf.py · conformal_rate.py · alarms.py · fallback.py
│   ├── explain/       reasons.py
│   ├── evaluation/    metrics.py · protocols.py · artefacts.py   # writes reports/metrics.json
│   ├── api/           app.py · schemas.py · service.py
│   └── stages/        baseline.py · train.py · conformal.py · fairness.py · monitor.py · report.py · bench.py
├── experiments/       stress_variants.py · inject_bugs.py
├── notebooks/         01_eda.ipynb · 02_results.ipynb      # thin: import from src/, outputs stripped
├── app/demo.py
├── models/            # git-ignored: model artefacts · manifest.json · monitor_state.json
├── reports/           metrics.json · monitoring.json · monitor_events.jsonl · tables/policy_grid.csv · figures/
│                      validation_report.md · model_card.md · reviews.md · data_contract.md
├── docs/              PROGRESS.md · DECISIONS.md
├── docker/Dockerfile  # python:3.11-slim, uv, non-root user, healthcheck
└── tests/             fixtures/make_fixture.py · test_*.py
```

---

## 7. Data

### Source and download

- **Source:** <https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022>
  - Paper: <https://arxiv.org/abs/2211.13358>
  - Docs and datasheet: <https://github.com/feedzai/bank-account-fraud>
  - Use **Base**, **Variant IV** and **Variant V**.
- **Download:**
  - List files: `kaggle datasets files sgpjesus/bank-account-fraud-dataset-neurips-2022`
  - Fetch one file at a time, so the three unused variants are not downloaded:
    `kaggle datasets download sgpjesus/bank-account-fraud-dataset-neurips-2022 -f "Base.csv" -p data/raw --unzip`
  - **Verified 2026-09-18.** `--unzip` does nothing when `-f` is used: each file arrives as
    a `.zip` with a URL-encoded name (`Variant%20IV.csv.zip`), and `make data` extracts it.
    The archive inside carries the correct name (`Variant IV.csv`).
  - File names contain spaces (e.g. `Variant IV.csv`). Normalised to snake_case in `data/interim/`.
  - The dataset publishes **six CSVs only** (Base and Variants I-V), 213-252 MB each.
    There are no parquet copies; `make data` writes the parquet itself, and zstd
    compression takes each variant from ~213 MB of CSV to ~60 MB (measured 2026-09-18).
- **Checksums.** On the first download, write `data/checksums.sha256`. After that, `make data` verifies against it.

### What the dataset contains

**Verified 2026-09-18 against the first load of Base, Variant IV and Variant V.**

- Each variant has 1,000,000 applications across 8 months (`month` 0–7). Monthly volume
  *declines*: 132,440 in month 0 down to 96,843 in month 7. Months are not equal sized.
- Fraud prevalence is 1.103% overall in Base, and by month: 1.133, 0.939, 0.875, 0.922,
  1.137, 1.183, 1.341, 1.475 (%). It rises through the test months.
- The protected attributes are age, income and employment status. Grouping age at 50 gives
  an 81.7 / 18.3 split in Base. Fraud is roughly three times more common in the older
  group (2.341% against 0.825%), which is why predictive equality is hard here.
- Variant IV and Variant V are the age-biased variants: both are 50.6% aged 50 or over.
  In Variant V the fraud rate is nearly equal across the two groups (1.118% against
  1.087%). All three variants carry exactly 1.103% fraud overall.
- **Variant V has 34 columns, not 32**: it ships two extra continuous columns, `x1` and
  `x2`. `make data` drops them, loudly, so every interim file has the same 32 columns.
- There are **no nulls anywhere**. Missing values arrive as negative numbers.
- In pandas, 1M rows × 32 columns takes about 0.5 GB.

### Expected columns

**Verified 2026-09-18.** All 32 column names below are exactly right. The frozen contract
in `src/triage/data/contract.py` carries the observed ranges, and `reports/data_contract.md`
is generated from it.

- **Label:** `fraud_bool`
- **Time:** `month`
- **Age and income:** `customer_age` (rounded to the decade); `income` (decile-like values, 0.1–0.9)
- **Categoricals:** `payment_type`, `employment_status`, `housing_status`, `source`, `device_os`
- **Binaries:** `email_is_free`, `phone_home_valid`, `phone_mobile_valid`, `has_other_cards`, `foreign_request`, `keep_alive_session`
- **Numerics:** `name_email_similarity`, `prev_address_months_count`, `current_address_months_count`, `days_since_request`, `intended_balcon_amount`, `zip_count_4w`, `velocity_6h`, `velocity_24h`, `velocity_4w`, `bank_branch_count_8w`, `date_of_birth_distinct_emails_4w`, `credit_risk_score`, `bank_months_count`, `proposed_credit_limit`, `session_length_in_minutes`, `device_distinct_emails_8w`, `device_fraud_count`

### Cleaning rules

- **Missing values are negative numbers** in several columns. **All confirmed 2026-09-18**,
  with the share of Base affected:
  - exactly `-1`, and never any other negative value: `prev_address_months_count` (71.3%),
    `current_address_months_count` (0.43%), `bank_months_count` (25.4%),
    `session_length_in_minutes` (0.20%), `device_distinct_emails_8w` (0.04%);
  - any negative value in `intended_balcon_amount` (74.3%, minimum −15.53, and no row is
    exactly −1).

  Add explicit `*_missing` flags for these in `features/sentinels.py`.
- **Two columns are negative without being missing.** Do not flag either as a sentinel:
  - `credit_risk_score` is negative for 1.44% of Base (minimum −170), including 488 rows at
    exactly −1. It is a score, not a count, so those are real values;
  - `velocity_6h` is negative for 44 rows of a million (minimum −170.6). That is not
    physically meaningful and is not documented anywhere: treat it as a published data
    quirk, flag it in the feature layer, and mention it in the validation report.
- **Zero-variance columns.** Confirmed: `device_fraud_count` is 0 for all 1,000,000 rows of
  Base. It is dropped from the features and recorded in `docs/DECISIONS.md`.
- **The contract** (`src/triage/data/contract.py`) is built from the first verified load and then frozen. It covers types, ranges, allowed category sets and nullability. Any later change needs an entry in `docs/DECISIONS.md`.

### Simulated days

BAF has no timestamps within a month. For monitoring:
- shuffle rows within each month using the global seed;
- cut each month into windows of **4,000 applications**, roughly one day's volume (1M
  applications over 8 months). A short remainder at the end of a month is dropped, because
  a part-window is not a day: Base yields 246 full windows over the eight months.

State this simulation assumption in the report.

---

## 8. Experiment protocol

### 8.1 Splits

**Paper protocol.** Used only to compare with published results.
- Train on months 0–5 and test on months 6–7.
- Set the threshold on the test set at FPR = 5%.

**Deployment protocol.** Used for everything else.
- Train on months 0–4.
- Split month 5 into three equal, disjoint parts, stratified by label and seeded:
  - `cal_prob`: fit probability calibration.
  - `cal_tune`: choose the α values and the 5%-FPR operating threshold.
  - `cal_conf`: compute conformal thresholds only. Never use it for any choice.
- Evaluate **month 6 and month 7 separately**. You may also report them pooled, but never only pooled.

**Champion tuning.** Train on months 0–3 and validate on month 4, with at most 30 trials. Then refit on months 0–4.

### 8.2 Models

- **B0.** Logistic regression with one-hot categoricals, standardised numerics and `class_weight="balanced"`.
- **B1.** LightGBM with default parameters on all features, *including* `customer_age`. This is the paper-comparable baseline.
- **Champion.** LightGBM *without* `customer_age`, lightly tuned. Always pass `seed`, `deterministic=True`, `force_row_wise=True` and a fixed `n_jobs`.
- **Calibration.** Compare no calibration, Platt and isotonic. Fit on `cal_prob` and choose by Brier score on `cal_tune`.
- **Fairness experiments:**
  - **M1**: the champion, with age dropped. This is usually not enough, because other fields act as proxies.
  - **M2**: FairGBM with `constraint_type="FPR"` and group `customer_age >= 50`, used in training only.
  - **M3**: fairlearn `ExponentiatedGradient` with `FalsePositiveRateParity`.
  - **M4**: policy only, relying on the review band.

  For all four, plot TPR@5%FPR against the FPR ratio.

### 8.3 Metric definitions

**Detection**
- **TPR@5%FPR.** Use `sklearn.metrics.roc_curve` and take the largest TPR with FPR ≤ 0.05.
- **Realised rates.** Report FPR and TPR for each test month at the threshold chosen on `cal_tune`.
- **ROC-AUC and PR-AUC.** For PR-AUC, use `average_precision_score`.

**Fairness**
- **FPR ratio (predictive equality).** `min(FPR_age≥50, FPR_age<50) / max(...)`, with FPR computed on genuine applicants only.
  - Also report FPR by 10-year age band.
  - Give a 95% percentile bootstrap CI: 1,000 resamples, stratified by (label, age group), with a fixed seed.
- **Relative likelihood**, as in the DWP assessment. Divide the rate for a group by the rate for the reference group, with a 95% CI. Flag anything outside 0.80–1.25 as "notable".

**Calibration**
- Report the Brier score.
- Report ECE with **15 equal-mass bins**. Equal-width bins are useless at about 1% prevalence.
- Report calibration-in-the-large: the mean predicted rate against the observed rate.
- Give each of these overall and by age group.

**Conformal**
- Fraud coverage: the share of fraud whose set contains "fraud".
- Genuine exclusion rate: the share of genuine applicants whose set lacks "legit".
- Approve, review, verify and empty-set shares.
- Report all of these per test month and per age band.

**Cost per 10,000 applications**
- Compute `review_cost × n_review + verify_cost × n_verify + Σ loss_proxy(missed fraud) + friction_cost × n_genuine_verified`.
- Always include a sensitivity table over cost ratios.

**Service**
- p50/p99 latency over 2,000 sequential requests, after 100 warm-up requests, on CPU only.
- Docker image size.

**Rounding.** Use 3 significant figures in prose, and keep full precision in artefacts.

### 8.4 Conformal policy (label-conditional split conformal)

Let `p(x)` be the calibrated fraud probability. On `cal_conf`, there are `n1` frauds and `n0` genuine applications:

```text
k1 = ceil((n1 + 1) * (1 - alpha_fraud))      k0 = ceil((n0 + 1) * (1 - alpha_legit))
tau_f = (n1 - k1 + 1)-th smallest p among cal_conf frauds     (-inf if k1 > n1)
tau_l = k0-th smallest p among cal_conf genuine               (+inf if k0 > n0)

"fraud" in C(x)  <=>  p(x) >= tau_f
"legit" in C(x)  <=>  p(x) <= tau_l

C = {legit}              -> approve
C = {legit, fraud} or {} -> review
C = {fraud}              -> verify
```

**Guarantees.** These assume exchangeability within each class.
- `P(fraud ∈ C | fraud) ≥ 1 − alpha_fraud`. So at most `alpha_fraud` of fraud is auto-approved.
- `P(legit ∉ C | genuine) ≤ alpha_legit`. This caps the genuine applicants sent to verify or to empty-set review.

**Implementation.**
- The threshold form above is equivalent to the textbook score form, whose nonconformity score is `1 − p_class`. Implement the threshold form.
- Take `tau_f` and `tau_l` directly from the sorted calibration probabilities. **Don't compute `tau_f = 1 − q̂`.** Isotonic calibration produces many tied values, and the float round-trip can flip decisions on those ties.

**Choosing α.**
1. Sweep `alpha_fraud` ∈ {0.05, 0.10, …, 0.60} × `alpha_legit` ∈ {0.01, 0.02, 0.05}, using thresholds computed on `cal_tune`.
2. Pick the pair with the highest fraud coverage whose review share fits the capacity.
3. Only then compute the final thresholds on `cal_conf`.

A 90% catch guarantee will probably need a very large review share. Show the whole trade-off curve; don't hide it.

**Coverage by age.** Report coverage by age band on the test months. If coverage falls short for older applicants, record that as a finding. Don't fix it with age-specific (Mondrian-by-age) thresholds in the policy, because that uses age at decision time (rule 4). An analysis-only comparison is allowed if it is labelled.

**Tests.**
- (a) The threshold form and the score form give identical decisions on probabilities drawn from a coarse grid (multiples of 0.001) with ties.
- (b) On synthetic exchangeable data, mean coverage over 200 seeded repeats is ≥ 1 − α − 0.01.
- (c) Edge cases: `k > n`, all-equal probabilities, and an empty calibration class, which must raise an error.

### 8.5 Monitoring

**Detectors.** All three need no labels.
1. **PSI** per feature and on the score, computed in DuckDB SQL (`monitoring/psi.sql`).
   - Reference: training months 0–4.
   - Numeric bins: the reference deciles.
   - Categoricals: each category, plus "other".
   - Smoothing: ε = 1e-4.
2. **Domain classifier.** A small LightGBM (50 trees) that separates a window from 4,000 reference rows sampled from `cal_conf`. Score it by 5-fold CV ROC-AUC.
3. **Conformal-rate test.** In each window, take the share of applications with `p > tau_l` and the share with `p >= tau_f`. Compare each with its rate on `cal_conf` using a binomial test, flagging at p < 0.001.

**Thresholds.** Calibrate each detector on 200 bootstrap "clean" windows drawn from `cal_prob ∪ cal_tune`, using the 99th percentile. For PSI, also report results against the rule of thumb of 0.10 (watch) and 0.25 (alert).

**Alarm logic.**
- **WATCH**: any detector exceeds its threshold in a window.
- **ALERT**: the same detector exceeds its threshold in 2 consecutive windows, or the score PSI is above 0.25.

**Fallback.** On ALERT:
- switch to the `policy.fallback` α preset, which sends more cases to review;
- append an event to `reports/monitor_events.jsonl`;
- update `models/monitor_state.json`, which the API reads.

**Experiments** (in `experiments/`)

- **Stress test.** Score the Base-trained champion on months 6–7 of Variant IV and of Variant V.
- **Injected bugs.** Each starts at window 5 of month 6:
  - (a) swap the two most frequent `employment_status` codes;
  - (b) mirror `income` using a lookup table over its observed values (lowest ↔ highest). Don't use `1 − x`, so the values stay exactly within the contract;
  - (c) set `phone_mobile_valid` to 1 for every row;
  - (d) multiply `income` by 10. This bug is outside the contract, so the contract must reject it both in batch validation and at the API. Test that it does.
- **What to report for each bug:**
  - detection delay (applications from the first bugged row to ALERT);
  - false alarms on clean windows;
  - model degradation (TPR at the deployment threshold before vs after the bug).
- Alarms on the untouched months 6–7 are natural drift. Report them separately, and don't call them false alarms.

---

## 9. Milestones and acceptance criteria

### Week 1: a baseline you can defend (tag `v0.1`)
- [ ] Scaffold the project: uv, Makefile, Hydra configs, pre-commit (ruff, nbstripout), and CI running `make lint test`.
- [ ] Run `make data` and write the checksums. Write the contract and make it pass. Add an EDA notebook showing fraud rate by month and by age band.
- [ ] Implement both protocols. Run B0 and B1 with bootstrap CIs.
- [ ] Add the fairness metrics (FPR ratio and age bands) and the cost/capacity frame.
- [ ] Write README v0 with a generated baseline table.

**Done when:**
- `make baseline` reproduces every number from a clean clone.
- B1's paper-protocol TPR@5%FPR is within a few points of the public reference. A third-party implementation with a time-based split reports 0.535 recall at 5% FPR and ROC-AUC 0.89, and the BAF paper reports an FPR ratio of about 0.3 for its best Base models. Use these as sanity checks only.

### Week 2: the differentiators (tag `v0.2`)
- [ ] Compare calibration methods and plot reliability curves, overall and by age group.
- [ ] Build the conformal policy (§8.4) with the α sweep. Report test-month coverage by month and by age band.
- [ ] Run fairness experiments M1–M4 and draw the trade-off plot.
- [ ] Build the monitor (§8.5) with thresholds calibrated on clean windows.
- [ ] Run the stress tests and injected bugs. Draft "What didn't work".

**Done when:**
- Month-6 fraud coverage is within ±1.5 points of the target.
- Every in-contract bug triggers an ALERT.
- The out-of-contract bug is rejected.
- The false-alarm rate on clean windows is recorded.

### Week 3: ship, document, rehearse (tag `v1.0`)
- [ ] Build the FastAPI service (§10), the Docker image and `make bench`.
- [ ] Complete the full test suite (§13) and get CI green.
- [ ] Build the Streamlit demo (§12). The owner records a 2-minute walkthrough.
- [ ] Write the two-page validation report and the model card. Log two outside reviews in `reports/reviews.md`; the owner arranges the reviewers.
- [ ] Finalise the README.

**Done when:**
- A fresh clone plus `make all` reproduces the README.
- CI is green.
- The report is no more than 2 pages.

### If you fall behind

**Cut in this order:**
1. Variant IV (keep Variant V).
2. The Streamlit demo (keep a notebook).
3. The reason-code stability check.
4. FairGBM (keep fairlearn, or just the review band).

**Never cut:** time-based splits, CIs, the conformal coverage check, the injected-bug test, or "What didn't work".

**Stretch goal, only after v1.0.** An LLM that drafts case notes for analysts. It may use only the structured fields and reason codes, with a check that every sentence maps to one of those fields.

---

## 10. API contract

**`POST /score`** scores one application.
- **Request body:** `{"application_id": str | null, "features": {...}}`.
- **Validation.** The features model is generated from the contract with `extra="forbid"`. It enforces ranges and category sets, and invalid input returns 422.
- **Age handling.** `customer_age` is accepted and logged for fairness monitoring, but dropped before scoring. Test this: changing it must not change the score.

The response looks like this (values are illustrative only):

```json
{
  "application_id": "demo-001",
  "risk_score": 0.0213,
  "decision": "review",
  "conformal_set": ["legit", "fraud"],
  "reasons": [
    {"feature": "name_email_similarity", "direction": "low",
     "text": "Name and email address look unrelated"}
  ],
  "model_version": "champion-<git_sha8>",
  "policy_version": "alpha_fraud=<a>,alpha_legit=<b>",
  "drift_status": "ok",
  "fallback_active": false
}
```

**Other endpoints:**
- `GET /health`
- `GET /version`, which returns the manifest
- `GET /monitor/status`, which reads `models/monitor_state.json`

**Behaviour:**
- **Startup.** Load artefacts once, then verify their hashes against `models/manifest.json`. Refuse to start on a mismatch.
- **Latency.** `?explain=false` skips SHAP for latency tests. The target p50 is under about 15 ms on CPU; report whatever you actually get.
- **Reasons are for analysts only.** No demo text should ever show them to an applicant.

---

## 11. Reason codes

- Use SHAP `TreeExplainer` on the *uncalibrated* LightGBM margin, and say so in the docs.
- Take the top 3 features pushing the score towards fraud.
- Write plain-English templates in `configs/reasons.yaml`, keyed by feature and direction. The direction is "high" or "low" relative to the training median. No ML or physics jargon.
- Optional: check how often the top reason stays the same across months.

---

## 12. Demo (Streamlit)

- `app/demo.py` reads only precomputed aggregates: `reports/tables/policy_grid.csv` and `reports/metrics.json`. The app does no training and no heavy computation.
- It is one screen:
  1. A review-capacity slider selects an α pair: the one with the best coverage within that capacity on `cal_tune`.
  2. The screen then shows month-6 and month-7 outcomes for that pair:
     - approve, review and verify shares;
     - fraud caught;
     - genuine applicants reviewed;
     - false-alarm rate by age group, with CIs;
     - expected cost.
- Use plain-English labels, and show a visible note that the data is synthetic.

---

## 13. Testing and quality

### Test data

Tests must run without the Kaggle data. `tests/fixtures/make_fixture.py` generates a small, seeded, BAF-shaped dataset with a known signal. Mark any test that needs the real data with `@pytest.mark.data`.

### Required tests

| Test file | What it checks |
|---|---|
| `test_conformal.py` | Everything in §8.4 |
| `test_psi.py` | Known-answer PSI on hand-made bins; identical distributions give 0 |
| `test_fairness.py` | FPR ratio on a hand-computed confusion matrix; the bootstrap is reproducible with a seed |
| `test_contract.py` | Rejects out-of-range values, unseen categories and the `income × 10` bug |
| `test_policy.py` | Set → decision mapping, including the empty set |
| `test_splits.py` | No month overlap; the `cal_*` parts are disjoint and stratified |
| `test_repro.py` | The same seed gives an identical hash of the scores |
| `test_api.py` | Returns 200 for valid input and 422 for invalid; scores don't change with age; drift status is present |
| `test_fairgbm_smoke.py` | FairGBM trains; skipped when FairGBM is unavailable |
| `test_regression.py` | Pinned metrics on the fixture stay within tolerance |

### Quality bar

- **Coverage:** at least 85% for `uncertainty/`, `fairness/`, `monitoring/` and `policy/`.
- **Style:**
  - type hints on public functions and short docstrings;
  - small, pure functions;
  - no logic in notebooks;
  - `ruff` formatting and non-strict `mypy`.
- **Definition of done for any change:**
  - `make lint test` passes;
  - any affected artefacts are regenerated with `make report`;
  - `docs/PROGRESS.md` is updated.

---

## 14. Writing rules (README, report, docs)

### Language

- Write plain English for a smart non-specialist. Avoid physics jargon ("surrogate", "regime", "Navier–Stokes").
- Explain conformal prediction once: *"a way to set thresholds so that a stated share of fraud is caught, as long as new applications look like the calibration data."*
- Never name a bank. Say "a UK retail bank".

### README

**Section order:**
1. One-line summary
2. Why it matters (UK context, with links)
3. How it decides (a diagram of the three bands)
4. Results (generated tables)
5. What didn't work (at least 3 real items)
6. What this does not show
7. How to run
8. Repo map
9. Validation report
10. Data licence notice

**"What this does not show" must cover:**
- the data is synthetic and not from the UK;
- there are no vulnerability labels, so no vulnerability analysis;
- age is used only for measurement and training constraints;
- the guarantees assume exchangeability;
- the costs are illustrative.

### Validation report

`reports/validation_report.md` must fit in 2 pages when exported, with these sections:
1. Purpose, users and proposed risk tier
2. Data and known gaps
3. Method and key choices
4. Performance: discrimination, calibration, and stability by month
5. Fairness: FPR ratio with CIs, and what each mitigation cost
6. Monitoring plan: metrics, thresholds, owner, fallback, and re-validation triggers
7. Findings rated High, Medium or Low, with recommendations

### Figures

- Label axes, with units.
- Show CIs where available.
- Put the month in the title.
- Use a colour-blind-safe palette.

### Context facts you may cite

Always include the link.
- **UK APP scam losses** were £576.4m in 2025 (up 19%), across 248,070 cases.
  Source: UK Finance, Annual Fraud Report 2026: <https://www.ukfinance.org.uk/news-and-insight/press-release/fraud-report-2026-press-release>
- **APP scam reimbursement.** Since 7 Oct 2024, victims can be reimbursed up to £85,000, with the cost split 50:50 between the sending and receiving firms.
  Summary: <https://www.hlc.com/en/publications/uk-app-fraud-what-in-scope-psps-need-to-know-about-the-new-mandatory-reimbursement-regime>
- **PRA SS1/23 model risk management principles**, in force since 17 May 2024.
  Source: <https://www.bankofengland.co.uk/prudential-regulation/publication/2023/may/model-risk-management-principles-for-banks-ss>
- **UK automated decision-making safeguards** under the Data (Use and Access) Act, in force since 5 Feb 2026.
  Summary: <https://www.cliffordchance.com/insights/resources/blogs/talking-tech/en/articles/2026/02/key-aspects-of-the-data--use-and-access--act-take-effect.html>
- **FCA Consumer Duty outcomes-monitoring review** (Jul 2026).
  Summary: <https://www.tlt.com/insights-and-events/insight/fca-raises-the-bar-on-consumer-duty-outcomes-monitoring-what-your-firm-needs-to-do-now>
- **A UK public-sector fairness template for a fraud model** (DWP, Jul 2025).
  Source: <https://assets.publishing.service.gov.uk/media/6876219388da2e5804bb6a17/UC_Advances_model_fairness_assessment_-_July_2025.pdf>

---

## 15. Session workflow

**Start of session**
1. Read `docs/PROGRESS.md` (create it if it's missing).
2. Run `git status` and `make test`.
3. Propose no more than three tasks that fit about 2.5 hours, and confirm them with the owner.

**During the session**
- Work in small increments.
- Use Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `exp:`).
- Log every experiment run to MLflow.
- Keep single runs under about 10 minutes on a laptop, and use `data.sample_frac` while developing.

**Stopping cleanly**
- Leave `make lint test` green.
- Put any unfinished work behind a flag or on a branch.

**End of session.** Update `docs/PROGRESS.md` with:
- the date;
- what changed;
- results, copied from artefacts along with each artefact's path;
- what didn't work;
- the next step;
- open questions for the owner.

Record any lasting choice in `docs/DECISIONS.md` as context → decision → consequences.

**Other rules**
- If `data/interim/` is missing, don't try to download the data without the owner's Kaggle credentials. Ask the owner to run `make data`.
- Don't start week-2 work until week 1's acceptance criteria pass, unless the owner asks.

---

## 16. References

- **The BAF paper:** Jesus et al., "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation", NeurIPS 2022. <https://arxiv.org/abs/2211.13358>
- **BAF docs and datasheet:** <https://github.com/feedzai/bank-account-fraud>
- **Third-party reference with a time-based split** (sanity check only): <https://github.com/muhammadayyass/bank-fraud-detection-fairness-audit>
- **A BAF study that used a random split**, useful as a contrast: <https://www.mdpi.com/2079-3197/13/12/290>
- **Libraries:**
  - FairGBM: <https://pypi.org/project/fairgbm/>
  - fairlearn: <https://pypi.org/project/fairlearn/>
  - MAPIE: <https://pypi.org/project/mapie/>
