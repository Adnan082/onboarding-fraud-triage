# Progress

Newest entry first. Each session records: what changed, results (copied from the
artefact, with its path), what didn't work, the next step, and open questions.

---

## 2026-09-21 — Going public: a licence, a generated summary, and a live demo

**What changed**
- `LICENSE` (MIT, the code only). The README's licence section now states what
  the data is published under -- CC BY-NC-ND 4.0 on the Kaggle page -- rather
  than telling the reader to go and look, and says plainly that no BAF data is
  committed here.
- A generated `headline` block at the top of Results: six lines built from
  `metrics.json` like every other table. Typed numbers introduced into the prose
  earlier in the session were removed; they broke rule 6 and would have drifted.
- A second demo tab that scores one editable application by posting to the
  running API, with the form built from the frozen contract (DECISIONS D23).
  `make demo` starts both processes. `tests/test_demo.py` checks every preset
  against the generated request model.
- `example_application()` moved from `stages/bench.py` to `api/schemas.py`, so
  the benchmark and the demo send the same starting payload.

**Results**
- The three demo presets land on the three different bands, scored through the
  live API: "Established, long settled" 0.010 approve `{legit}`; "Contract
  midpoint" 0.053 review `{legit, fraud}`; "Thin file, shared device" 0.588
  verify `{fraud}`. Reason codes on all three.
- Clean-clone `make all`: exit 0, and twenty headline figures identical to this
  working copy (see the reproducibility note above).
- 324 tests, 306 of them data-free. Coverage gate green.

**What didn't work**
- **The repository is not public yet, and should not be until the licence
  question is answered.** The dataset is CC BY-NC-**ND** -- no derivatives --
  and while no BAF data is committed, aggregate output computed from it is.
  Whether that is a derivative work is a legal question, and rule 9 says flag
  those rather than answer them. The LICENSE and the README are ready either way.
- `test_the_same_window_measures_the_same_twice` compared floats exactly and
  failed under load when a PSI sum reassociated in its last bit. Fixed to compare
  to twelve places, as `test_repro.py` already did. The test was wrong, not the
  runner.
- A first attempt at the demo imported `requests`, which is not in the pinned
  stack and only arrives transitively through Streamlit. Switched to `httpx`,
  which is declared.

**Next step**
- Answer the licence question, then make the repository public.
- Set the alpha targets, review capacity and cost parameters; every headline
  number is conditional on them.
- Optional: split the dependency set so the service image stops carrying mlflow,
  streamlit, matplotlib, kaggle, duckdb, mapie and fairlearn; add a matched
  with-age comparator so M1 is an ablation again.

**Open questions for the owner**
- The licence reading above.
- Two outside reviews for `reports/reviews.md`, and the 2-minute demo walkthrough
  -- now easier, because the demo shows a single application being decided.

---

## 2026-09-20/21 — MLflow, champion tuning, Docker, and the report inside two pages

**What changed**
- `src/triage/tracking.py`: MLflow logging for every stage, wired through
  `runtime.stage_run`. Params, scalar metrics, config hash, git SHA and the raw
  data checksums. Sampled runs are tagged `sampled=True`.
- `champion.tune()` and `write_tuning_trials()`: a seeded 30-trial random search
  on months 0-3 scored on month 4, with the configured parameters run alongside
  as trial -1. The winner is pasted into `configs/model/champion.yaml`; tuning is
  off by default (DECISIONS D22).
- The whole chain re-run on the tuned champion: `baseline train conformal
  fairness monitor bench report`.
- `docker/Dockerfile` copies `README.md` before the second `uv sync`. The image
  builds and serves.
- `evaluation/validation.py`: prose compressed and `estimated_pages()` added, so
  the two-page cap in section 14 is now a test rather than an intention (D21).
- `tests/test_tuning.py` (5 tests), `tests/test_monitor_runner.py` (7) and a
  page-budget test. 315 tests, 306 of them data-free.
- `make coverage` gates the four section 13 packages at 85% and CI runs it.
  `monitoring/runner.py` had 0% coverage and pulled the package to 81.9%, under
  the bar; it is now at 100% and the four together are at 95.13%.

**Results** (from `reports/metrics.json` unless noted)
- **Tuning** (`reports/tables/tuning_trials.csv`): best trial 0.5785 TPR@5%FPR on
  month 4 against the incumbent's 0.5661. 29 of 30 trials lost. Spread 0.500 to
  0.579. Winner: 1,000 trees, lr 0.02, 15 leaves, min_child 200, colsample 0.9,
  subsample 0.8.
- **Champion, month 6 / month 7**: ROC-AUC 0.8906 / 0.8951, TPR 0.5724 / 0.5840
  at FPR 0.0650 / 0.0550, Brier 0.01211 / 0.01292. Calibration: platt.
- **Baselines, deployment protocol**: B0 ROC-AUC 0.8812 / 0.8862, B1 0.8871 /
  0.8901. B1 pooled under the paper protocol: ROC-AUC 0.8873, TPR@5%FPR 0.5202
  (95% CI 0.5010-0.5389), against the public reference's 0.89 and 0.535.
- **Conformal policy**: fraud coverage 0.5814 / 0.5910 against a 0.55 target;
  genuine exclusion 0.0160 / 0.0130 against a 0.01 promise -- still breached, on
  both months.
- **Fairness, FPR ratio**: M1 drop age 0.4334 / 0.4429; M2 FairGBM 0.4454 /
  0.4728 at TPR 0.5414 / 0.5630; M3 fairlearn 0.1922 / 0.2525 at TPR 0.0379 /
  0.0420; M4 policy only 0.4358 / 0.4439 at TPR 0.5814 / 0.5910. B1, which sees
  age, sits at 0.3309 / 0.3373.
- **Monitor** (`reports/monitoring.json`): 200 clean windows give 6 watch, 0
  alert. Thresholds: score PSI 0.01337, worst-feature PSI 3.968, domain AUC
  0.5198, conformal-rate -log p 1.830.
- **Injected bugs, on a quiet stream**: all four detected. `income_x10` by the
  contract at window 0 (never scored); `mirror_income`, `swap_employment` and
  `all_mobile_valid` by the monitor one window after injection. `mirror_income`
  costs 6.76 points of TPR while undetected.
- **Latency**, 2,000 sequential requests after 100 warm-up: p50 6.42 ms over HTTP
  without reason codes (target 15 ms), p99 8.90 ms; 148.84 ms with them, p99
  255.83 ms. Slower with SHAP than before, because the tuned champion has twice
  the trees to walk.
- **Docker image**: 767 MB compressed, 3.35 GB unpacked
  (`reports/metrics.json:service.docker`).
- **Validation report**: 952 words, about 1.98 estimated pages, 5 findings
  (2 High, 2 Medium, 1 Low).

**What didn't work**
- **MLflow's file store refuses to run.** MLflow 3 puts `./mlruns` in maintenance
  mode and raises rather than starting a run, so the first tuning run logged
  nothing and said so in a warning. Moved to a local SQLite store (D20).
- **The tuning win is inside the noise.** Month 4 holds 1,452 frauds, so the
  standard error on a TPR near 0.57 is about 1.3 points and the 1.2-point gain
  sits inside it. Taken because both held-out months moved the same way, not
  because one month proved it.
- **Tuning cost a controlled comparison.** The champion and B1 no longer share
  hyper-parameters, so "dropping age costs nothing" is now a comparison of two
  fitted models rather than an ablation. Both README claims were reworded to say
  so. A matched with-age comparator would restore it; not added, because M1-M4
  are what section 8.2 specifies.
- **Three pinned regression metrics moved** and were re-pinned: ROC-AUC 0.9175 ->
  0.9295, TPR@5%FPR 0.5897 -> 0.6154, fraud coverage 0.7692 -> 0.7179, approve
  share 0.8560 -> 0.8880, all on the fixture. The reason is the parameter change
  and nothing else.
- **The Docker image reports two sizes that differ by four times.** `docker image
  inspect` gives 766,642,640 bytes and `docker images` prints 3.35GB for the same
  image: the first is the compressed content a pull downloads, the second is what
  it occupies unpacked. `bench` recorded only the first, which would have left a
  reviewer checking with `docker images` and finding a number that disagreed. It
  now records both and the README prints both.
- **Most of that 3.35 GB is dependencies the service never loads.** Importing
  `triage.api.app` pulls in lightgbm, pandera and scikit-learn and nothing else
  heavy: mlflow, streamlit, matplotlib, kaggle, duckdb, mapie and fairlearn are
  all installed into the image and never used by it (shap loads lazily, only for
  `?explain=true`). The 2.37 GB `uv sync` layer is where that goes. Not acted on
  -- splitting the dependency set is section 4's pinned stack and the owner's
  call -- but it is the obvious way to shrink the image.
- **No generated table had ever been committed.** The blanket `*.csv` ignore that
  keeps row-level data out of git was also swallowing `reports/tables/`, so
  `policy_grid.csv`, `policy_outcomes.csv`, `fairness_tradeoff.csv` and the new
  `tuning_trials.csv` were missing from every clone. That broke the README's
  links to them and meant the demo, which reads `policy_grid.csv`, could not run
  from a clean clone without re-running the pipeline. `.gitignore` now excepts
  them, and they are aggregates -- the largest is 16 KB.

**Reproducibility check**
- A fresh `git clone` gets 107 tracked files and runs `make setup && make lint &&
  make test` green with no data at all.
- **The full criterion is closed.** A second clean clone ran `make all` end to
  end -- Kaggle download, contract, baseline, train, conformal, fairness,
  monitor, report, bench, tests -- and exited 0. Twenty headline figures were
  compared against this working copy and **all twenty are identical**: ROC-AUC,
  realised TPR, Brier and FPR ratio for both test months; fraud coverage and
  genuine exclusion for both; both conformal thresholds to nine decimal places;
  the calibration choice; the config hash; and all four monitor thresholds.
  Section 9's "a fresh clone plus `make all` reproduces the README" now has
  evidence behind it rather than an intention.

**Next step**
- Run `make all` from a clean clone with Kaggle credentials, to close the other
  half of the v1.0 criterion.
- Decide whether to split the dependency set so the service image stops carrying
  mlflow, streamlit, matplotlib, kaggle, duckdb, mapie and fairlearn.
- Consider a matched with-age comparator so M1's claim is an ablation again.

**Open questions for the owner**
- The alpha targets and cost parameters remain placeholders and are load-bearing
  for the headline claim.
- Kaggle licence check before the repo goes public.
- Two outside reviews for `reports/reviews.md`, and the 2-minute demo walkthrough.

---

## 2026-09-19 — Monitor, mitigations, API, and the reporting layer

**What changed**
- The drift monitor: `domain_clf`, `conformal_rate`, `alarms`, `fallback`,
  `runner`, and `stages/monitor.py`.
- Fairness mitigations M1-M4 (`models/fair.py`, `stages/fairness.py`).
- The scoring API: request schema generated from the contract, artefact loading
  with hash verification, SHAP reason codes, `stages/bench.py`.
- The reporting layer: `evaluation/validation.py` (the two-page report),
  `evaluation/model_card.py`, `evaluation/figures.py` (four figures),
  `policy/cost.py`, and six more README tables.
- The Streamlit demo, reading only precomputed artefacts.
- `test_api`, `test_repro`, `test_regression` are now real: **294 tests, zero
  skipped**.

**Results** (all from `reports/metrics.json` and `reports/monitoring.json`)
- **Detector calibration**: 200 clean windows give 4.0% watch and 0% alert,
  against 3.9% expected for four detectors at the 99th percentile.
- **The monitor found the conformal breach without labels**: month 6 window 0
  shows a crossing rate of 2.43% against the 1.34% expected (p ~ 4e-9) and a
  domain AUC of 0.758.
- **Injected bugs, on a quiet stream** (DECISIONS D17): all three in-contract
  bugs reach ALERT one window after injection. `income_x10` is rejected by the
  contract and never scored. Detection cost while undetected: -12.6 points of TPR
  for `swap_employment`, -8.3 for `mirror_income`, -0.4 for `all_mobile_valid`.
- **Mitigations, month 6**: M1 drop age TPR 0.5690 / ratio 0.439; M2 FairGBM
  0.5414 / 0.445; M3 fairlearn **0.0379** / 0.192; M4 policy only **0.5897** /
  **0.450**.
- **Latency**: p50 7.25 ms over HTTP without reason codes (target 15 ms), 151 ms
  with them. Scoring alone is 5.42 ms, so the HTTP layer costs under 2 ms.
- Validation report: 1,310 words, 5 findings (2 High, 2 Medium, 1 Low).

**What didn't work**
- **Both model-level fairness mitigations failed.** FairGBM's FPR constraint did
  not separate from simply dropping age. fairlearn's ExponentiatedGradient
  collapsed to flagging almost nobody and was still the least equal of the four.
  The policy-only option won on both axes.
- `psi_feature_max` is effectively dead weight: `velocity_4w` trends structurally
  and swamps it (DECISIONS D18).
- The first benchmark run reported a cheerful p50 of 0.66 ms while the service was
  returning 503 to every request. It now refuses to measure a failing request.
- Section 8.5's bug experiment could not attribute detection on month 6, because
  month 6 alarms on its own drift (DECISIONS D17).

**Known gap**
- The validation report is ~1,310 words with 20 table rows, which exports to
  roughly two and a half pages against section 14's two-page cap. Further cuts
  would remove findings rather than prose.

**Next step**
- `make docker` (needs Docker Desktop running), then the README's final pass.
- Two open decisions: MLflow logging, and whether to run the 30-trial champion
  tuning.

**Open questions for the owner**
- The alpha targets and cost parameters remain placeholders and are now
  load-bearing for the headline claim.
- Kaggle licence check before the repo goes public.
- Two outside reviews for `reports/reviews.md`, and the 2-minute demo walkthrough.

---

## 2026-09-18 (late) — Champion, calibration, and the conformal policy

**What changed**
- `models/calibrate.py`: none / Platt / isotonic, fitted on `cal_prob`, chosen on
  `cal_tune`, plus the reliability curve stored for the figure.
- `models/champion.py`: fit (refusing to run if `use_age` is true), save with a
  hashed manifest, and `load_fitted`, which every later stage uses so they all
  score with the same artefacts the API serves.
- `stages/train.py` and `stages/conformal.py`.
- `policy/sweep.py`: the alpha grid, capacity selection, coverage by age band.
- Five more README tables (calibration, policy, trade-off curve, coverage by age)
  and two more "What didn't work" entries.

**Results** (from `reports/metrics.json`, sections `champion` and `policy`, and
`reports/tables/policy_grid.csv`)
- Calibration on `cal_tune`: Platt **0.010507** Brier / **0.00120** ECE; isotonic
  0.010566 / 0.00259; uncalibrated 0.010542 / 0.00353. The uncalibrated model
  predicts **0.865%** against an observed **1.184%**.
- Champion (no age) vs B1 (age): ROC-AUC **0.8862 vs 0.8871** on month 6 and
  0.8893 vs 0.8901 on month 7. **Dropping age costs essentially nothing.**
- FPR ratio, month 6: champion **0.439 (0.418-0.460)** against B1's 0.331
  (0.316-0.346). M1 helps materially and still leaves older applicants stopped
  about 2.3x as often (FPR 0.119 vs 0.052).
- Calibration is markedly worse for the older group: Brier 0.0177 vs 0.0090 on
  `cal_tune`, and on month 7 ECE 0.00948 vs 0.00135 with the model
  under-predicting fraud for over-50s by 0.9 points.
- Policy chosen on `cal_tune`: `alpha_fraud=0.45`, `alpha_legit=0.01` — the best
  coverage inside the placeholder 5% review capacity. Thresholds on `cal_conf`:
  tau_fraud 0.038648, tau_legit 0.132032, from 470 frauds.
- **Fraud coverage held**: 0.5897 (month 6) and 0.5833 (month 7) against a 0.55
  target. **Genuine exclusion did not**: 0.0158 and 0.0115 against a 0.01 promise.
- Trade-off curve: 55% coverage costs a 4.6% review share; 90% costs **30.7%**;
  95% costs 44.5%. On 100,000 applications that is 4,564 reviews against 30,703.
- Coverage by age band, month 6: 0.458 for 20-29 rising to 0.712 for 60-69, with
  review share 2.9% against 10.9%. The policy is age-blind; the burden is not.
- Tests: **215 passed, 3 skipped**; lint and mypy clean.

**What didn't work**
- The genuine-exclusion guarantee was breached on both test months. Exchangeability
  between month 5 and months 6-7 does not hold, which is the monitor's whole
  reason for existing.
- Fraud coverage over-delivered by ~4 points, outside the +/-1.5 point acceptance
  criterion in section 9. `cal_conf` holds only 470 frauds, so the threshold is a
  noisy order statistic.
- Two tests I wrote first were wrong, not the code: one fixture was not actually
  miscalibrated, and one asserted a review-share relationship that only holds when
  the score distributions overlap the way the real ones do. Both now assert what is
  true unconditionally.

**Next step**
- The monitor (section 8.5): the three detectors, thresholds calibrated on clean
  windows, and the injected-bug experiments. The genuine-exclusion breach above is
  the thing it needs to catch.

**Open questions for the owner**
1. **The alpha targets are now load-bearing.** The policy currently promises to
   catch 55% of fraud because that is what a 5% review capacity buys. Both numbers
   are placeholders. The trade-off table is the thing to look at before deciding.
2. `git_sha` is still `unknown`, and it is now stamped into `models/manifest.json`
   and reported by the API as `model_version: champion-unknown`.
3. Unchanged: cost parameters, the `proposed_credit_limit` proxy, DECISIONS D4,
   and the stale Windows copy.

---

## 2026-09-18 (night) — First real results: B0 and B1 on the full million

**What changed**
- `features/sentinels.py` and `features/encode.py`: the six real sentinels flagged
  and blanked, the two look-alikes deliberately left alone (D10), one-hot and
  LightGBM encodings both pinned to the frozen contract's category levels.
- `models/baselines.py`: B0 and B1, each owning its own preprocessing.
- `evaluation/protocols.py`: both protocols end to end, with fairness measured in
  every slice.
- `evaluation/artefacts.py`, `evaluation/tables.py`, `stages/report.py`: the rule-6
  path. Stages write `reports/metrics.json`; `make report` renders the README
  tables from it; nothing else may write a number into the README.
- README: four generated tables (data, detection, fairness, age bands) and the
  first three real "What didn't work" entries.

**Results** (all from `reports/metrics.json`, section `baselines`)
- **B1 matches the published references in section 9.** Pooled paper protocol:
  ROC-AUC **0.8873**, TPR@5%FPR **0.5202**. Pooled deployment: ROC-AUC **0.8883**,
  TPR@5%FPR **0.5309**. The third-party reference is 0.535 recall at 5% FPR and
  ROC-AUC 0.89. Week 1's acceptance criterion is met.
- **B0 is barely behind B1.** Month 6, deployment: ROC-AUC 0.8812 vs 0.8871,
  TPR@5%FPR 0.5021 (0.476-0.528) vs 0.5124 (0.488-0.538). The intervals overlap
  heavily.
- **The cal_tune threshold does not transfer.** Target 5% FPR; realised: B1 month 6
  6.19% (+24%), month 7 5.67% (+13%); B0 month 6 5.48% (+10%), month 7 3.30%
  (-34%).
- **Fairness reproduces the paper.** FPR ratio at the deployment threshold: B1
  0.331 (0.316-0.346) month 6 and 0.337 (0.320-0.356) month 7; B0 0.302 and 0.273.
  Section 9 cites about 0.3 for the paper's best Base models.
- FPR by 10-year band (B1, month 6) climbs monotonically: 1.76% at 10-19, 4.23% at
  30-39, 12.3% at 50-59, 17.2% at 60-69. Bands above 70 hold too few genuine
  applicants to read, and the README marks them as such.
- Runtime: the full baseline stage takes ~5 minutes, almost all bootstrap.
- Tests: **195 passed, 3 skipped**; lint and mypy clean.

**What didn't work**
- Gradient boosting barely beat logistic regression — about one point of TPR, with
  overlapping intervals. Recorded in the README.
- A threshold set on month 5 does not hold on months 6 and 7, in either direction.
- `with_model` could not merge a second model config into Hydra's composed one,
  because the models have different parameter sets; it replaces the node instead.

**Next step**
- `make train`: the champion without age, plus the calibration comparison
  (none / Platt / isotonic, chosen by Brier on `cal_tune`), and `models/manifest.json`.
- Then the conformal policy, which is where the threshold-transfer problem above
  gets its real answer.

**Open questions for the owner**
- Unchanged: capacity and cost parameters, the `proposed_credit_limit` proxy, the
  alpha targets, DECISIONS D4, the stale Windows copy, and the missing git commit
  (artefacts still record `git_sha=unknown`).

---

## 2026-09-18 (evening) — Full dataset, and the move to WSL2

**What changed**
- Downloaded the remaining three variants: all six BAF CSVs are now in
  `data/raw` (1.3 GB). The pipeline still reads Base, Variant IV and Variant V.
- Benchmarked the real workload before choosing where to train (DECISIONS D12).
- Moved the project to `~/projects/onboarding-fraud-triage` in WSL2 Ubuntu-22.04:
  uv installed, project copied (106 files, 1.2 MB), Kaggle token copied, data
  copied to ext4, interim parquet rebuilt there.
- Fixed `config_hash` so the provenance key survives moving the checkout
  (DECISIONS D11), with two tests in `tests/test_scaffold.py`.

**Results**
- Timings on this laptop (Core Ultra 7 258V, 8 cores, 32 GB): load 1M rows 0.2 s;
  LightGBM champion 500 trees on 675,666 rows **7.6 s**; score 108,168 rows 0.46 s;
  B0 logistic regression 1.9 s. Extrapolated: 30-trial tuning ~4 min, M3 fairlearn
  ~6 min, all 246 monitoring windows ~4 s. No cloud needed.
- A throwaway champion-shaped fit scored ROC-AUC 0.886 and TPR@5%FPR 0.519 on
  month 6, against the public references of 0.89 and 0.535 in section 9. Not a
  reportable number -- no calibration, no protocol -- but the pipeline looks sound.
- In WSL: **150 passed, 4 skipped** data-free, 3 passed with `-m data`; lint and
  mypy clean. Contract passes for all three variants.
- The checksums written on Windows verify unchanged after the copy to ext4, so the
  files are byte-identical across the move.
- `config_hash` is now `1f6a7be2` on both Windows and WSL, verified on each.

**What didn't work / findings**
- **`libgomp1` is missing from the distro**, so LightGBM *and* FairGBM fail to
  import in WSL. It needs `sudo apt install libgomp1`, which needs the owner. Note
  the Dockerfile already installs it.
- `make` is not installed in the distro either; same one-line fix.
- The config hash was machine-dependent, caught only because a new test called it
  outside a Hydra run (D11).

**Next step**
- Once `libgomp1` is in: confirm FairGBM imports, then `features/sentinels.py` and
  `features/encode.py`, then B0 and B1 under both protocols.

**Open questions for the owner**
- Unchanged: capacity and cost parameters, the `proposed_credit_limit` proxy, the
  alpha targets, and DECISIONS D4.
- The Windows copy under OneDrive is now stale. Safe to delete once you are happy
  with WSL.
- Still no git commit, so every artefact records `git_sha=unknown`.

---

## 2026-09-18 (later) — Real data in, contract frozen

**What changed**
- Kaggle credentials in place; downloaded Base, Variant IV and Variant V only
  (`-f` per file, ~680 MB of CSV rather than the full 1.4 GB).
- `src/triage/data/load.py`: checksum write/verify, and CSV to parquet in DuckDB
  (62 MB per variant with zstd, ~1 second each).
- `src/triage/data/contract.py`: **contract v1.0 frozen** from the first verified
  load, plus `reports/data_contract.md` generated from it.
- `tests/test_contract.py` is real: 26 tests, all on the fixture, plus three
  `@pytest.mark.data` tests against the real variants.
- Corrected CLAUDE.md section 7 against the load, and updated the Makefile `data`
  target to match what the Kaggle CLI actually does.
- `src/triage/runtime.py`: the seeding/provenance helper, so `data.load` and
  `data.contract` run as Hydra apps on the same footing as the stages.

**Results**
- 3,000,000 rows converted; `data/checksums.sha256` written for all three files.
- Contract v1.0: **base, variant_iv and variant_v all pass**
  (`reports/data_contract.md`).
- Tests: **148 passed, 4 skipped** data-free; **3 passed** with `-m data`. Lint and
  mypy clean.
- Real split sizes (deployment protocol): train 675,666 (months 0-4);
  cal_prob 39,782 / cal_tune 39,771 / cal_conf 39,770; test month 6 = 108,168 and
  month 7 = 96,843. Simulated days: **246 full 4,000-row windows**.
- **`cal_conf` holds only 470 frauds.** That is the resolution limit on the
  conformal thresholds, and it needs saying next to any coverage number.
- Base fraud prevalence 1.103% overall, rising 0.875% (month 2) to 1.475%
  (month 7). Monthly volume *declines*, 132,440 to 96,843.
- Age: 18.3% of Base is 50 or over, and fraud is ~3x more common in that group
  (2.341% against 0.825%). Variants IV and V are 50.6% older, and in Variant V the
  two groups have nearly equal fraud rates (1.118% against 1.087%).

**What didn't work / findings**
- `kaggle datasets download --unzip` is silently ignored when `-f` is used: files
  arrive as `.zip` with URL-encoded names. Both the Makefile and section 7 now say so.
- DuckDB rejects a bound parameter as a `COPY ... TO` target; the paths are inlined
  and escaped instead.
- **Variant V has 34 columns**, not 32: two extra continuous columns `x1` and `x2`.
  Dropped at the interim step (DECISIONS D9).
- **Negative does not always mean missing** (DECISIONS D10). `credit_risk_score` is
  negative for 1.44% of Base including 488 rows at exactly -1, and those are real
  scores. `velocity_6h` is negative for 44 rows in a million, which is not
  physically meaningful and is documented nowhere; it gets a flag and a line in the
  validation report.
- An exact-range contract would have rejected both variants, which drift past
  Base's minima and maxima on ten columns. Bounds now carry documented headroom
  (DECISIONS D8), while closed domains such as `income` stay exact, which is what
  keeps the `income x 10` rejection working.

**Next step**
- `features/sentinels.py` (the six sentinel columns, and the two that only look
  like sentinels) and `features/encode.py`, then B0 and B1 under both protocols.

**Open questions for the owner**
- Unchanged: capacity and cost parameters, the `proposed_credit_limit` proxy, the
  alpha targets, and DECISIONS D4.
- New: `data/` now holds 827 MB inside a OneDrive-synced folder. Worth excluding
  from sync, or moving the project off OneDrive.

---

## 2026-09-18 — Conformal, splits, metrics, fairness and PSI (all data-free)

**What changed**
- `tests/fixtures/make_fixture.py`: the seeded, BAF-shaped generator every
  data-free test now runs on (DECISIONS D5).
- `src/triage/uncertainty/conformal.py`: label-conditional split conformal,
  written from scratch (rule 7). Threshold form, taken straight off the sorted
  calibration probabilities, plus a score-form reference implementation used only
  by the tests.
- `src/triage/data/split.py`: both protocols, the three stratified `cal_*` parts,
  and the simulated 4,000-application days.
- `src/triage/evaluation/metrics.py`: TPR@5%FPR, the matching threshold, realised
  rates per month, ROC-AUC/PR-AUC, Brier, equal-mass ECE and
  calibration-in-the-large. `ece_equal_mass` lives in `models/calibrate.py`.
- `src/triage/fairness/`: FPR, FPR ratio, FPR by group and by 10-year band,
  relative likelihood with the DWP 0.80-1.25 "notable" flag, and the stratified
  percentile bootstrap.
- `src/triage/monitoring/psi.py` + `psi.sql`: PSI in DuckDB, with a pandas
  implementation as the definition and a test that the two agree.
- `src/triage/policy/decide.py`: `decide_many` and `band_shares`.
- Tests: `test_conformal.py`, `test_splits.py`, `test_fairness.py`, `test_psi.py`,
  `test_policy.py` are now real, plus a new `test_metrics.py`.

**Results** (local, `uv run pytest -m "not data"`)
- **125 passed, 5 skipped**, up from 23 passed / 10 skipped. `ruff check`,
  `ruff format --check` and `mypy src` clean.
- Coverage of the modules implemented so far: `uncertainty/conformal.py` 95%,
  `fairness/bootstrap.py` 100%, `fairness/metrics.py` 97%, `monitoring/psi.py` 98%,
  `data/split.py` 95%, `evaluation/metrics.py` 96%, `policy/decide.py` 94%.
  The 85% package bar is met for `uncertainty/` and `fairness/`; `monitoring/` and
  `policy/` are still pulled down by their unimplemented modules.
- Conformal coverage check: mean fraud coverage over 200 seeded repeats clears
  `1 - alpha - 0.01` at both alpha pairs tested, and the genuine exclusion rate
  stays under `alpha_legit + 0.01`.
- PSI: the DuckDB path matches the pandas definition to 1e-12 on five fixture
  columns, and identical distributions give exactly 0 in both.
- No model results. `reports/metrics.json` still does not exist.

**What didn't work / what to watch**
- On the hand-built 20-row fairness slice the FPR-ratio point estimate is 0.25 and
  its 95% bootstrap interval is the whole range [0, 1]. Recorded as a test
  (`test_a_tiny_slice_gives_a_useless_interval`). The lesson carries into the
  report: narrow 10-year age bands at ~1% prevalence will not support a claim, and
  the band table must show intervals and say when they are uninformative.
- The MAPIE marginal cross-check is not written yet. It is a nice-to-have from
  section 4, not one of the section 13 requirements, and it needs a fitted
  estimator, so it waits for the training code.

**Next step**
- Still blocked on data for `make data` / the contract. Next data-free steps, in
  order: `features/sentinels.py` and `features/encode.py`, then B0/B1 and the
  champion on the fixture, which unblocks `test_repro.py`, `test_regression.py`
  and the API tests.

**Open questions for the owner**
- Unchanged from 2026-09-17: capacity and cost parameters, the
  `proposed_credit_limit` proxy, the alpha targets, Kaggle credentials, and
  DECISIONS D4.

---

## 2026-09-17 — Scaffold

**What changed**
- Initialised the repository (`git init`, no commit yet) and laid out the
  structure in CLAUDE.md section 6. Renamed `CLAUDE_1.md` to `CLAUDE.md`, the
  name the layout expects.
- Tooling: `pyproject.toml` with the pinned stack and the `fairgbm` Linux-only
  extra, `Makefile` with every target in section 5, `.pre-commit-config.yaml`
  (ruff, ruff-format, nbstripout, hygiene hooks), `.github/workflows/ci.yml`
  running `lint` + data-free tests, `docker/Dockerfile`.
- Hydra configs for data, features, model (logreg / lgbm / champion / fairgbm /
  fairlearn_eg), calibration, policy, monitor, plus `configs/reasons.yaml`.
- Hydra logs to the console only (`configs/hydra/job_logging/console.yaml`), so
  stages never leave `*.log` files in the repository.
- Module skeletons across `src/triage/`, `experiments/`, `app/demo.py`, and the
  test files from section 13.
- Implemented for real: `triage.policy.decide.decide`, `triage.config`,
  `triage.seeding`, `/health`, `/version`, `/monitor/status`, and
  `tests/test_scaffold.py` + `tests/test_policy.py`.

**Results**
- No model results yet. `reports/metrics.json` does not exist.
- Scaffold gate, run locally on Windows with uv 0.10.6 / CPython 3.11.9:
  - `uv lock` resolves 167 packages; `uv sync` installs cleanly.
  - `ruff check` + `ruff format --check` + `mypy src` (49 files): clean.
  - `pytest -m "not data"`: **23 passed, 10 skipped** (the skips are the section 13
    suites waiting on their modules).
  - `python -m triage.stages.baseline` composes the config, seeds, stamps
    provenance, then stops at its TODO -- as does an override run
    (`data.sample_frac=0.1 model=lgbm`), which changes the config hash and fires
    the "development run" warning.

**What didn't work**
- The stack as written did not install. `numpy>=2,<3` + `shap==0.51.0` dragged
  numba back to 0.53.1 (2021), whose llvmlite will not build on Python 3.11.
  Fixed by capping numpy at `<2.4` and constraining `numba>=0.61`; see DECISIONS
  D4, which needs the owner's sign-off under rule 11.

**Next step**
- Week 1, item 2: run `make data` (needs the owner's Kaggle credentials), write
  `data/checksums.sha256`, confirm the column list and the sentinel rules against
  the datasheet, then freeze the pandera contract.

**Open questions for the owner**
1. Review capacity and every cost parameter in `configs/policy/default.yaml` are
   placeholders tagged `# ASSUMPTION`. What should they be?
2. Is `proposed_credit_limit` an acceptable proxy for the loss on a missed fraud?
3. What are the alpha targets, and the fallback alpha preset?
4. Kaggle credentials: `make data` cannot run without them.
5. DECISIONS D4 changes two pins in CLAUDE.md section 4 (numpy cap, numba floor).
   It was needed to make the environment install at all -- please confirm.
