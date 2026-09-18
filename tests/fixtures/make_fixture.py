"""Generate a small, seeded, BAF-shaped dataset with a known signal.

The real Kaggle data is never needed for the fast test suite. Any test that does
need ``data/interim/`` is marked ``@pytest.mark.data``.

What this reproduces from the real thing (CLAUDE.md section 7):

- the documented columns, in the documented order;
- eight months, with fraud prevalence drifting between 0.85% and 1.5%;
- ``customer_age`` rounded to the decade, split roughly 80/20 at 50;
- negative sentinels for missing values, and one all-zero column
  (``device_fraud_count``) so the zero-variance check has something to find;
- a planted signal, so a model has something real to learn, and a deliberate
  age-linked proxy in ``credit_risk_score``, so the fairness work has a gap to
  measure even though the champion never sees age;
- a mild shift in months 6-7, standing in for natural drift.

It is not a substitute for the real data: the relationships are simple and the
prevalence is planted, so no reported number may ever come from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Skewed on purpose: the injected bug "swap the two most frequent codes" needs a
# well-defined top two.
CATEGORY_LEVELS: dict[str, list[str]] = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["windows", "macintosh", "linux", "x11", "other"],
}

CATEGORY_WEIGHTS: dict[str, list[float]] = {
    "payment_type": [0.38, 0.30, 0.17, 0.10, 0.05],
    "employment_status": [0.42, 0.24, 0.13, 0.09, 0.06, 0.04, 0.02],
    "housing_status": [0.30, 0.22, 0.17, 0.12, 0.09, 0.06, 0.04],
    "source": [0.93, 0.07],
    "device_os": [0.34, 0.26, 0.21, 0.13, 0.06],
}

AGE_DECADES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
AGE_WEIGHTS = [0.03, 0.22, 0.31, 0.24, 0.11, 0.05, 0.025, 0.010, 0.005]

# Roughly the real spread, month by month.
MONTHLY_PREVALENCE = [0.0110, 0.0095, 0.0085, 0.0092, 0.0104, 0.0121, 0.0138, 0.0150]

# Columns whose missing value arrives as -1, with the rate at which one is planted.
MINUS_ONE_RATES: dict[str, float] = {
    "prev_address_months_count": 0.72,
    "current_address_months_count": 0.02,
    "bank_months_count": 0.25,
    "session_length_in_minutes": 0.03,
    "device_distinct_emails_8w": 0.01,
}

COLUMN_ORDER = [
    "fraud_bool",
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "payment_type",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "employment_status",
    "credit_risk_score",
    "email_is_free",
    "housing_status",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "source",
    "session_length_in_minutes",
    "device_os",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "device_fraud_count",
    "month",
]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _standardise(values: np.ndarray) -> np.ndarray:
    floats = values.astype(float)
    spread = floats.std()
    return (floats - floats.mean()) / spread if spread > 0 else floats * 0.0


def _intercept_for_prevalence(logits: np.ndarray, target: float) -> float:
    """Bisect the intercept, so the planted prevalence lands on target."""
    low, high = -30.0, 30.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if _sigmoid(logits + mid).mean() < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def make_fixture(n_rows: int = 4000, seed: int = 20260917, n_months: int = 8) -> pd.DataFrame:
    """Return a BAF-shaped frame of ``n_rows`` applications spread over ``n_months``.

    Deterministic given ``seed``.
    """
    if n_rows < n_months:
        raise ValueError("n_rows must be at least n_months")

    rng = np.random.default_rng(seed)
    month = np.repeat(np.arange(n_months), n_rows // n_months)
    month = np.concatenate([month, np.full(n_rows - month.size, n_months - 1)])
    n = month.size
    late = (month >= 6).astype(float)  # months 6-7 drift a little

    age = rng.choice(AGE_DECADES, size=n, p=AGE_WEIGHTS)
    older = (age >= 50).astype(float)

    frame = pd.DataFrame({"month": month, "customer_age": age})

    for column, levels in CATEGORY_LEVELS.items():
        weights = np.array(CATEGORY_WEIGHTS[column], dtype=float)
        frame[column] = rng.choice(levels, size=n, p=weights / weights.sum())

    # Income is decile-like and rises a little with age.
    income_step = rng.integers(1, 10, size=n) + older * rng.integers(0, 2, size=n)
    frame["income"] = np.clip(income_step, 1, 9) / 10.0

    frame["name_email_similarity"] = rng.beta(2.0, 2.2, size=n)
    frame["days_since_request"] = rng.exponential(0.9, size=n).round(6)
    frame["zip_count_4w"] = rng.integers(10, 6700, size=n)
    frame["velocity_6h"] = rng.normal(5200 + 500 * late, 2600, size=n).clip(0, 17000).round(6)
    frame["velocity_24h"] = rng.normal(5000, 1400, size=n).clip(1300, 9500).round(6)
    frame["velocity_4w"] = rng.normal(4800, 900, size=n).clip(2800, 7000).round(6)
    frame["bank_branch_count_8w"] = rng.integers(0, 2400, size=n)
    frame["date_of_birth_distinct_emails_4w"] = rng.poisson(1.2, size=n).clip(0, 40)
    frame["proposed_credit_limit"] = rng.choice(
        [200.0, 500.0, 1000.0, 1500.0, 2000.0], size=n, p=[0.32, 0.30, 0.20, 0.12, 0.06]
    )
    frame["device_fraud_count"] = 0  # all-zero in the real data too: zero variance

    # The age proxy. This is why dropping age on its own (M1) is not enough.
    credit_risk = rng.normal(110, 70, size=n) + 38 * older - 25 * frame["income"].to_numpy()
    frame["credit_risk_score"] = credit_risk.clip(-200, 400).round(0)

    for column in ("email_is_free", "keep_alive_session"):
        frame[column] = rng.binomial(1, 0.52, size=n)
    frame["phone_home_valid"] = rng.binomial(1, 0.42, size=n)
    frame["phone_mobile_valid"] = rng.binomial(1, 0.89, size=n)
    frame["has_other_cards"] = rng.binomial(1, 0.23, size=n)
    frame["foreign_request"] = rng.binomial(1, 0.03, size=n)

    frame["prev_address_months_count"] = rng.integers(0, 380, size=n)
    frame["current_address_months_count"] = rng.integers(0, 430, size=n)
    frame["bank_months_count"] = rng.integers(0, 32, size=n)
    frame["session_length_in_minutes"] = rng.gamma(2.0, 3.5, size=n).clip(0, 110).round(6)
    frame["device_distinct_emails_8w"] = rng.integers(0, 4, size=n)
    frame["intended_balcon_amount"] = rng.gamma(1.5, 8.0, size=n).clip(0, 115).round(6)

    # The planted signal, on features the champion is allowed to see.
    logits = (
        -1.90 * _standardise(frame["name_email_similarity"].to_numpy())
        + 0.95 * _standardise(frame["credit_risk_score"].to_numpy())
        + 0.55 * _standardise(frame["velocity_6h"].to_numpy())
        + 0.45 * _standardise(frame["date_of_birth_distinct_emails_4w"].to_numpy())
        + 0.40 * _standardise(frame["device_distinct_emails_8w"].to_numpy())
        - 0.85 * _standardise(frame["income"].to_numpy())
        - 0.75 * frame["has_other_cards"].to_numpy()
        + 0.50 * frame["email_is_free"].to_numpy()
        + 0.60 * frame["foreign_request"].to_numpy()
    )

    fraud = np.zeros(n, dtype=int)
    for month_id in range(n_months):
        mask = month == month_id
        target = MONTHLY_PREVALENCE[month_id % len(MONTHLY_PREVALENCE)]
        intercept = _intercept_for_prevalence(logits[mask], target)
        fraud[mask] = rng.binomial(1, _sigmoid(logits[mask] + intercept))
    frame["fraud_bool"] = fraud

    # Sentinels go in last, so they do not disturb the planted signal.
    for column, rate in MINUS_ONE_RATES.items():
        frame.loc[rng.random(n) < rate, column] = -1
    negative_balcon = rng.random(n) < 0.74
    frame.loc[negative_balcon, "intended_balcon_amount"] = -rng.random(int(negative_balcon.sum()))

    return frame[COLUMN_ORDER].reset_index(drop=True)


if __name__ == "__main__":  # pragma: no cover - convenience for regenerating fixtures
    make_fixture().to_parquet("tests/fixtures/baf_fixture.parquet")
