# Model card — onboarding fraud triage champion

> **Draft skeleton.** Populated from `models/manifest.json` and
> `reports/metrics.json` once `make train` has run.

## Model details
_LightGBM, trained without `customer_age`. Version, config hash and git SHA come
from the manifest._

## Intended use
_Triage of online bank-account applications into approve / review / verify, inside
a portfolio project. Not a production system. Reason codes are for analysts only
and are never shown to an applicant._

## Out of scope
_Any automatic decline. Any claim about a named bank. Any use on real customer
data._

## Training data
_BAF Base, months 0-4 (deployment protocol). Synthetic, not UK data._

## Metrics
_Discrimination, calibration and conformal coverage, per test month._

## Fairness
_FPR ratio by age group with CIs, and FPR by 10-year band._

## Caveats
_Coverage guarantees hold only while new data is exchangeable with the calibration
data. Costs are illustrative parameters._
