# Decisions

Each entry: **context → decision → consequences**. Lasting choices only; session
notes belong in `PROGRESS.md`.

---

## D1 — 2026-09-17 — Repository layout and toolchain

**Context.** The project has to be reproducible from a clean clone by a reviewer
who has never seen it, and CLAUDE.md fixes both the layout and the dependency
versions.

**Decision.** Follow CLAUDE.md section 6 exactly. Python 3.11 with `uv`, a
hatchling `src/` layout, Hydra for config, and a Makefile as the single entry
point for every stage. `fairgbm` is an optional extra with a
`sys_platform == 'linux'` marker, imported lazily.

**Consequences.** `make` is not available on plain Windows, so development
happens in WSL2 or in the Docker image. Every stage is runnable on its own via
`uv run python -m triage.stages.<name>`.

---

## D2 — 2026-09-17 — scikit-learn pinned to 1.5.2

**Context.** `fairgbm==0.9.14` (November 2022) passes a scikit-learn argument
that newer versions removed; it fails on 1.9.1 and trains on 1.5.2.

**Decision.** Pin `scikit-learn==1.5.2` for the whole project.

**Consequences.** Any scikit-learn upgrade needs `tests/test_fairgbm_smoke.py`
re-run first. If FairGBM is eventually dropped, the pin can be relaxed.

---

## D3 — 2026-09-17 — Placeholder policy parameters are marked, not guessed

**Context.** Review capacity, the cost parameters and the alpha targets are the
owner's to set (CLAUDE.md section 3), but the config has to load today.

**Decision.** `configs/policy/default.yaml` carries plausible placeholders, every
one tagged `# ASSUMPTION`, and no result may be published from them.

**Consequences.** Once the owner sets real values, this file changes and every
policy artefact is regenerated.

---

## D4 — 2026-09-17 — numpy capped at <2.4, with a numba resolver constraint

**Needs the owner's sign-off** (rule 11: the pinned stack is section 4 of CLAUDE.md).

**Context.** CLAUDE.md pins "numpy 2.x". Written as `numpy>=2,<3`, the universal
resolution picked numpy 2.4.6, and `shap==0.51.0` requires `numba` with no lower
bound, so the resolver walked numba back to **0.53.1 (March 2021)**. Its
`llvmlite==0.36.0` refuses to build on Python 3.11 (`only versions >=3.6,<3.10
are supported`), and `uv sync` failed outright.

**Decision.** Two changes, both narrowing rather than adding:
- `numpy>=2,<2.4` — the newest numpy that current numba supports.
- `[tool.uv] constraint-dependencies = ["numba>=0.61"]` — a resolver constraint,
  not a new dependency; numba is still pulled in only by shap.

Resolved versions: numpy 2.3.5, numba 0.62.1, llvmlite 0.45.1, with
`scikit-learn==1.5.2` untouched.

**Consequences.** `uv sync` installs cleanly on Python 3.11. If shap is ever
dropped, both constraints can go. Revisit the numpy cap when numba supports 2.4.

---

## D5 — 2026-09-18 — The seeded fixture is the substrate for every data-free test

**Context.** The Kaggle data needs the owner's credentials and is git-ignored, but
the test suite has to run on a clean clone and in CI.

**Decision.** `tests/fixtures/make_fixture.py` generates a BAF-shaped frame from a
seed: the documented columns in the documented order, eight months, prevalence
drifting 0.85%-1.5%, `customer_age` rounded to the decade and split ~80/20 at 50,
the negative sentinels, `device_fraud_count` all zero, a planted signal, and a
deliberate age proxy in `credit_risk_score`. Months 6-7 drift slightly.

**Consequences.** Tests assert real behaviour rather than mocks, and the age proxy
means M1 ("just drop age") can be shown to be insufficient without the real data.
The fixture's relationships are simple and its prevalence is planted, so no
reported number may ever come from it; anything that needs the real data is marked
`@pytest.mark.data`.

---

## D6 — 2026-09-18 — Band shares sum to 1; the empty set is reported inside review

**Context.** Section 8.3 asks for approve, review, verify *and* empty-set shares,
but an empty set routes to review (rule 5), so four shares summing to more than 1
would be ambiguous in a table.

**Decision.** `coverage_report` returns `approve_share + review_share +
verify_share == 1`, with `empty_share` as a diagnostic *inside* `review_share`.

**Consequences.** Report tables can be read as a breakdown of 100% of
applications, and the empty-set count is still visible, since a rising empty share
is an early sign that the calibration no longer fits.

---

## D7 — 2026-09-18 — Simulated days drop the part-window at the end of a month

**Context.** BAF has no timestamps within a month, so section 7 simulates days by
shuffling within the month and cutting into 4,000-application windows. Months do
not divide evenly by 4,000.

**Decision.** `simulated_windows` keeps whole windows only and drops the remainder:
a part-window is not a day, and detector thresholds calibrated on 4,000-row windows
do not transfer to a 1,200-row one.

**Consequences.** A few applications per month never appear in a monitoring window.
This must be stated in the report alongside the simulation assumption itself.

---

## D8 — 2026-09-18 — The contract carries headroom over Base's observed ranges

**Context.** The contract is frozen from the first verified load of Base. But
Variant IV and Variant V drift slightly past Base's exact minima and maxima on ten
numeric columns (`velocity_6h` reaches 16,802 against Base's 16,716;
`zip_count_4w` 6,830 against 6,700). An exact-range contract would reject the
stress tests for no good reason.

**Decision.** Two kinds of bound:
- **closed domains get exact sets** — binaries, the five category level sets, the
  nine age decades, months 0-7, and `income` on its 0.1-0.9 decile grid;
- **open-ended numerics get the observed Base range plus documented headroom**,
  rounded to plain numbers (e.g. `velocity_6h` allowed `[-500, 20000]`).

**Consequences.** All three variants validate. The `income x 10` bug still fails,
because `income` is a closed domain — which is the check the monitoring experiment
depends on. A subtler corruption inside the headroom will not be caught by the
contract; that is the monitor's job, which is the point of having both.

---

## D9 — 2026-09-18 — Variant V's extra columns are dropped at the interim step

**Context.** Variant V ships **34 columns**: the 32 Base has, plus two continuous
columns `x1` and `x2` that Base and Variant IV do not have. The champion is
trained on Base, so the stress test must score the same 32 columns.

**Decision.** `make data` selects exactly the contracted columns and logs a warning
naming anything dropped. A *missing* contracted column is an error, not a warning.

**Consequences.** All three interim files have identical shape, and the contract
stays strict, so an unexpected column in a production feed is still a breach.
Anyone wanting `x1`/`x2` must go back to `data/raw/`.

---

## D10 — 2026-09-18 — Negative does not always mean missing

**Context.** Section 7 said missing values arrive as negative numbers. The first
load shows that is true for six columns and false for two others.

**Decision.**
- Sentinels, flagged in `features/sentinels.py`: `prev_address_months_count` (71.3%
  of Base), `bank_months_count` (25.4%), `current_address_months_count` (0.43%),
  `session_length_in_minutes` (0.20%), `device_distinct_emails_8w` (0.04%), each
  exactly `-1`; and `intended_balcon_amount` (74.3%), any negative value.
- **Not sentinels:** `credit_risk_score` is negative for 1.44% of Base including
  488 rows at exactly `-1`, but it is a score, not a count, so those are real
  values. `velocity_6h` is negative for 44 rows in a million (minimum -170.6),
  which is not physically meaningful and is documented nowhere.

**Consequences.** Treating `credit_risk_score == -1` as missing would corrupt a
feature the model leans on heavily. The `velocity_6h` negatives get their own flag
and a line in the validation report as a published data quirk, not a fix.

---

## D11 — 2026-09-18 — The config hash covers the config as authored

**Context.** `config_hash` is stamped on every artefact as the provenance key that
says "these two runs used the same settings". It hashed the *resolved* config,
which had two faults: it embedded absolute paths, so the same settings hashed
differently on Windows (`fcdbc7ca`) and in WSL (`2087c0ec`); and resolving needs
`HydraConfig` to be set, so it raised outside a stage run -- in a test, a
notebook, or anything using a plain `compose()`.

**Decision.** Hash the config as authored (`resolve=False`), with the
machine-specific `paths` and `mlflow` sections removed.

**Consequences.** The same settings now hash to `1f6a7be2` on both Windows and
WSL, verified. A change that affects results still changes the hash, because seeds,
alphas and overrides are stored values rather than interpolations. Two configs
that resolve alike but are written differently now hash differently, which is the
correct reading of "the same configuration".

---

## D12 — 2026-09-18 — WSL2 is the working environment

**Context.** The Makefile assumes a POSIX shell, `make` is not on Windows, and
FairGBM (M2) ships a Linux `.so`. Measured on this laptop, the whole workload is
small: a 500-tree LightGBM fit on 675,666 rows takes 7.6 seconds on 8 cores, so
there is no case for cloud training.

**Decision.** The project lives at `~/projects/onboarding-fraud-triage` inside
WSL2 Ubuntu-22.04, on the Linux filesystem rather than `/mnt/c`. Training runs
locally. The Windows copy under OneDrive is retired.

**Consequences.** `make`, FairGBM and the POSIX shell all work, and 1.3 GB of
data is out of OneDrive's sync. `libgomp1` must be installed in the distro:
LightGBM and FairGBM both fail to import without it. Anything cloud-based would
have cost the reproducibility story (`make all` from a clean clone) and made the
section 8.3 latency benchmark meaningless.

---

## D13 — 2026-09-18 — B0 imputes; LightGBM does not

**Context.** Once a sentinel is blanked to `NaN`, B0's logistic regression cannot
fit: scikit-learn's `LogisticRegression` rejects missing values. LightGBM handles
them natively and splits on missingness itself.

**Decision.** The B0 pipeline imputes numerics with the **training** median
(`features.encoding.impute_numerics`), after the `*_missing` flag has already been
added. LightGBM gets the `NaN` untouched.

**Consequences.** No information is lost: the flag tells B0 that the value was
missing, and the median keeps the coefficient interpretable. The two models see
slightly different inputs, which is worth remembering when comparing them —
though on the first run they landed within a point of each other anyway.

---

## D14 — 2026-09-18 — Confidence intervals on detection metrics too

**Context.** Section 8.3 specifies bootstrap CIs for fairness metrics. Week 1 also
asks for the baselines to carry CIs, without saying how many resamples.

**Decision.** One `bootstrap` block in `configs/config.yaml` (1,000 resamples, 95%
level) drives every published interval: TPR@5%FPR stratified by label, and the FPR
ratio stratified by (label, age group).

**Consequences.** A full baseline run takes about five minutes, nearly all of it
bootstrap. It is worth it: on month 6 the B0 and B1 intervals overlap heavily,
which is the whole reason the "LightGBM barely beat logistic regression" finding
can be stated honestly rather than as a point-estimate horse race. `stratum_members`
is hoisted out of the resampling loop, which cut the cost substantially without
changing a single number.

---

## D15 — 2026-09-18 — Platt over isotonic, chosen on cal_tune

**Context.** Three candidates were fitted on `cal_prob` and compared on `cal_tune`
by Brier score, as section 8.2 requires. The uncalibrated model predicts a 0.865%
fraud rate against an observed 1.184% — it ranks well and is badly wrong about
magnitude, which matters because the decision policy reads probabilities.

**Decision.** Platt scaling, on the log-odds rather than the raw probability.
Brier 0.010507 against isotonic's 0.010566 and 0.010542 uncalibrated; ECE 0.00120
against 0.00259 and 0.00353.

**Consequences.** The margins are thin, so this is worth re-checking whenever the
model changes. Feeding a sigmoid raw probabilities that are already near zero
leaves it almost no range to work with, hence the log-odds; isotonic's many tied
values are exactly the case the conformal threshold form was written to survive.

---

## D16 — 2026-09-18 — An infeasible capacity is reported, not quietly relaxed

**Context.** `select_within_capacity` can be asked for more fraud coverage than
the configured review capacity can absorb. The tempting behaviour is to return the
closest affordable pair.

**Decision.** It raises, naming the cheapest pair on the grid and what it would
cost. The stage catches that, logs it, falls back to the configured placeholder
alphas, and records `infeasible` in `reports/metrics.json`.

**Consequences.** "This policy cannot be staffed at this capacity" reaches the
owner as a finding rather than disappearing into a silently weaker policy. With
the current placeholder capacity (5% review, 2% verify) the grid does have
feasible pairs, so this path is exercised only by tests for now.

---

## D17 — 2026-09-19 — Bug detection is measured on a quiet stream, not only on month 6

**Context.** Section 8.5 injects each bug from window 5 of month 6 and measures
the delay to ALERT. Empirically month 6 is *already* in ALERT before any bug
lands: it has drifted from month 5 on its own, and 26 of its 27 windows alarm
with no bug at all. A delay measured there cannot be attributed to the bug.

**Decision.** Run both. The section 8.5 experiment is reported as specified, with
`already_alerting_before_bug` recorded against it, and a second experiment injects
the same bugs into a stream drawn from month 5, which is quiet by construction
(18 of 19 windows OK, 0 alerts).

**Consequences.** The attributable numbers are the ones worth quoting: all three
in-contract bugs reach ALERT one window after injection, which is the floor the
two-consecutive-windows rule allows. The out-of-contract bug is rejected by the
contract and never scored at all.

---

## D18 — 2026-09-19 — `psi_feature_max` is kept but known to be weak

**Context.** The worst-feature PSI detector has a clean-window threshold of 3.97.
A PSI of 4 is far beyond the 0.25 "major shift" rule of thumb, and the maximum is
`velocity_4w` in all 51 test windows: a four-week velocity trends structurally
between months, so it swamps the maximum and leaves the detector with almost no
headroom to signal anything else.

**Decision.** Keep it, report the limitation, and lean on the other three
detectors. Excluding structurally time-varying features would need a rule for
which those are, and inventing that rule post hoc on the test months is exactly
the kind of choice the protocol exists to prevent.

**Consequences.** Three effective detectors rather than four. Worth revisiting
with a reference window that moves with time, rather than a fixed training-month
reference.

---

## D19 — 2026-09-19 — Reason codes are optional per request

**Context.** The latency benchmark puts a scored request at p50 7.25 ms without
reason codes and 151 ms with them. SHAP walks every tree for every request, so
explanations cost roughly twenty times the entire latency budget in section 10.

**Decision.** `?explain=false` skips them, and the benchmark reports both paths
rather than a single headline number.

**Consequences.** The section 10 target is met for scoring and missed by an order
of magnitude for explanation. A real deployment would compute reason codes out of
band, for the applications a human is going to look at anyway, rather than on
every request.
