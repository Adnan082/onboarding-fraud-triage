# Model validation report

**Model:** onboarding fraud triage — application scoring and decision policy  
**Version:** `champion-76a45a60`  
**Generated:** 2026-09-21 from `reports/metrics.json` (config `8c54f36c`, commit `76a45a60`)  
**Status:** portfolio project on public synthetic data, not a production model.

## 1. Purpose, users and proposed risk tier

Scores online bank-account applications and routes each to **approve**, **human
review** or **extra verification**; the worst it can do to a customer is ask for
checks. Users are fraud operations and model risk. **Proposed tier: high** -- it
gates access to a payment account, measures a protected characteristic, and falls
under the Data (Use and Access) Act. No automatic decline lowers the severity of
one error, not the tier.

## 2. Data and known gaps

Bank Account Fraud (NeurIPS 2022): 1,000,000 applications over eight
months, 1.10% fraud. Split by month only -- train 0-4,
calibrate on month 5 in three parts, report 6 and 7 separately.

Known gaps, worst first. The data is **synthetic and not UK data**, so no rate
here transfers to a UK population. There is **no vulnerability information**, so
no vulnerability analysis is possible. There are **no within-month timestamps**,
so a 'day' is 4,000 shuffled applications: an assumption. The **costs are
illustrative**. No saving is claimed, in pounds or otherwise.

## 3. Method and key choices

LightGBM without `customer_age`, calibrated on one third of month 5 and selected
on another. Bands come from label-conditional split conformal prediction: two
thresholds, fitted on a third part of month 5 that nothing else touches, turn a
probability into a set of plausible labels, and the set picks the band. Three
choices worth testing: **age is excluded and used only to measure**, at almost no
cost in detection; **thresholds are never set on evaluation data**; and **the
conformal calibration part is used once**.

## 4. Performance

| Month | ROC-AUC | Fraud caught | False alarms | Brier | FA under 50 | FA 50+ | Ratio (95% CI) |
|---|---|---|---|---|---|---|---|
| 6 | 0.891 | 57.2% | 6.5% | 0.01211 | 5.2% | 12.0% | 0.433 (0.415-0.455) |
| 7 | 0.895 | 58.4% | 5.5% | 0.01292 | 4.7% | 10.7% | 0.443 (0.418-0.47) |

Calibration **platt**, on held-out data. Uncalibrated it predicts a 0.88% fraud rate against 1.18% observed: it ranks well and is
wrong about magnitude, which matters: the policy reads probabilities.
Discrimination holds on both months; the false alarm rate does not (finding 1).

## 5. Fairness

The last three columns are predictive equality, on genuine applicants only; a
ratio of 1.00 would mean both age groups stopped equally often. Age measures
the model, which never sees it.

| Mitigation, month 6 | Fraud caught | False alarms | Ratio |
|---|---|---|---|
| `m1_drop_age` | 57.2% | 6.5% | 0.433 |
| `m2_fairgbm` | 54.1% | 6.2% | 0.445 |
| `m3_fairlearn_eg` | 3.8% | 0.1% | 0.192 |
| `m4_policy_only` | 58.1% | 6.8% | 0.436 |

Both months are in the README. Neither model-level mitigation beat dropping
age, and M3 equalised by flagging almost nobody yet was still least equal.

## 6. Monitoring plan

Label-free detectors on 4,000-application windows, at the 99th percentile of 200 clean ones:

| Detector | Threshold | What it sees |
|---|---|---|
| Score PSI | 0.01337 | the distribution of risk scores moving |
| Worst feature PSI | 3.97 | any single input moving |
| Domain classifier AUC | 0.52 | the window being distinguishable from the reference at all |
| Conformal rate test | 1.83 | the policy's own crossing rates drifting from calibration |

**Alarms.** Watch when any detector exceeds its threshold; alert when one does
so twice running, or score PSI passes 0.25. **Fallback.** On alert the policy
tightens its alphas, sending more work to humans; the event is logged, read
on every request, and never cleared automatically.
**Owner.** Fraud operations own the queue, model risk the thresholds.
**Re-validation:** any alert; any change to the model, its calibration or the
alphas; a new calibration month; or a year.

## 7. Findings

**1. The genuine-applicant guarantee did not hold out of sample** — *High*. At most 1.0% of genuine applicants should be verified; 1.6% were, on both test months. Exchangeability with the calibration month fails: prevalence rises, scores move. *Recommendation:* re-derive thresholds on a recent month, on the alert; never quote the figure without its condition.

**2. Fraud coverage is looser than intended** — *Medium*. Coverage came in +4.1 points against a 55% target, outside the ±1.5 point tolerance. Over-delivery is safe, but the thresholds are loose: they rest on 470 frauds, so each is an order statistic from a small sample. *Recommendation:* size the calibration set by the frauds in it, not the applications.

**3. Burden falls unevenly across age bands** — *High*. Fraud coverage runs 33.3% to 76.7% across age bands, the review share 2.3% to 11.3%. The policy is age-blind, so this is the model showing through one pair of thresholds. *Recommendation:* do not correct this with age-specific thresholds -- that needs legal sign-off. Report it; price the alternatives with the mitigations.

**4. Calibration is worse for older applicants** — *Medium*. Expected calibration error is 0.00332 for age>=50 against 0.00134 for the best-served group. Thresholds come from pooled probabilities, so a less well calibrated group inherits a weaker guarantee, invisibly. *Recommendation:* report coverage by age band beside the headline guarantee, every time.

**5. Detector thresholds behave as designed on clean windows** — *Low*. Across 200 windows from the calibration month, 6 raised a watch and 0 an alert -- close to what a 99th-percentile threshold on four detectors implies, so the false-alarm rate is understood, not assumed. *Recommendation:* re-calibrate them whenever the model or the calibration month changes.

