# Model validation report — onboarding fraud triage

> **Draft skeleton.** Every number comes from `reports/metrics.json` and
> `reports/tables/` via `make report`. Nothing here is typed by hand. Must fit two
> pages when exported.

## 1. Purpose, users and proposed risk tier

_Who uses the model, for what decision, and the risk tier proposed under PRA
SS1/23. Note that there is no automatic decline._

## 2. Data and known gaps

_BAF is synthetic, generated from an unnamed bank's applications; it is not UK
data. Eight months, ~1% fraud. No timestamps within a month, so "days" are
simulated as seeded 4,000-application windows. No vulnerability labels._

## 3. Method and key choices

_Time-based splits only. Champion excludes `customer_age`. Probability calibration
chosen by Brier score on `cal_tune`. Label-conditional conformal thresholds fitted
on `cal_conf` alone._

## 4. Performance

_Discrimination, calibration and stability, month 6 and month 7 reported
separately._

## 5. Fairness

_FPR ratio by age with 95% bootstrap CIs, FPR by 10-year band, relative
likelihood against the reference group, and what each mitigation (M1-M4) cost in
detection._

## 6. Monitoring plan

_Detectors, thresholds, owner, fallback behaviour and re-validation triggers._

## 7. Findings

| # | Rating | Finding | Recommendation |
|---|---|---|---|
| | | _pending_ | |
