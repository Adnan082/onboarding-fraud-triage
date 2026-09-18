"""FairGBM (M2).

Required checks (CLAUDE.md section 13):
- FairGBM trains on the fixture;
- skipped with a clear message when FairGBM is unavailable (it is Linux-only).

This is a smoke test of the *dependency*, not of M2 itself. It exists because
`fairgbm==0.9.14` was last released in November 2022 and passes a scikit-learn
argument that newer versions removed: it fails on scikit-learn 1.9.1 and trains on
1.5.2, which is why that pin exists (CLAUDE.md section 4). If someone bumps
scikit-learn, this test is the canary that fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from triage.models.fair import fairgbm_available

pytestmark = pytest.mark.skipif(
    not fairgbm_available(),
    reason="FairGBM is unavailable: it ships a Linux .so, so it is expected to be missing "
    "on Windows and macOS. The M2 experiment skips with the same message.",
)

NUMERIC_FEATURES = [
    "income",
    "name_email_similarity",
    "credit_risk_score",
    "velocity_6h",
    "velocity_24h",
    "zip_count_4w",
    "date_of_birth_distinct_emails_4w",
    "device_distinct_emails_8w",
    "has_other_cards",
    "email_is_free",
    "foreign_request",
    "phone_mobile_valid",
]


@pytest.fixture(scope="module")
def training_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Features, labels, and the age constraint group. Age is never a feature."""
    from tests.fixtures.make_fixture import make_fixture

    frame = make_fixture(n_rows=8000, seed=20260917)
    features = frame[NUMERIC_FEATURES].astype(float)
    labels = frame["fraud_bool"].to_numpy()
    # Training-time constraint group only, never a model input (rule 4).
    group = (frame["customer_age"] >= 50).astype(int).to_numpy()
    return features, labels, group


def _fit(features: pd.DataFrame, labels: np.ndarray, group: np.ndarray, seed: int = 20260917):
    from fairgbm import FairGBMClassifier

    model = FairGBMClassifier(
        constraint_type="FPR",
        n_estimators=30,
        num_leaves=15,
        min_child_samples=50,
        random_state=seed,
        n_jobs=2,
        verbose=-1,
    )
    model.fit(features, labels, constraint_group=group)
    return model


def test_fairgbm_trains_and_scores(training_data) -> None:
    """The dependency works end to end on the pinned scikit-learn."""
    features, labels, group = training_data
    model = _fit(features, labels, group)

    scores = model.predict_proba(features)[:, 1]
    assert scores.shape == (len(features),)
    assert np.isfinite(scores).all()
    assert 0.0 <= scores.min() <= scores.max() <= 1.0
    assert scores.std() > 0, "a constant score means the fit did not learn anything"


def test_fairgbm_is_seeded(training_data) -> None:
    """Two fits with the same seed give the same scores (rule 10)."""
    features, labels, group = training_data
    first = _fit(features, labels, group).predict_proba(features)[:, 1]
    second = _fit(features, labels, group).predict_proba(features)[:, 1]
    assert np.array_equal(first, second)


def test_the_constraint_group_is_required(training_data) -> None:
    """FairGBM takes the group at fit time, and age never becomes a feature."""
    features, labels, group = training_data
    assert "customer_age" not in features.columns

    from fairgbm import FairGBMClassifier

    model = FairGBMClassifier(constraint_type="FPR", n_estimators=5, verbose=-1)
    with pytest.raises(TypeError):
        model.fit(features, labels)  # no constraint_group


def test_scikit_learn_is_the_pinned_version() -> None:
    """A reminder in the right place, not a hard gate.

    The pin exists for this library. If scikit-learn moves and the tests above
    still pass, the pin can be revisited; if they fail, this tells you why.
    """
    import sklearn

    if sklearn.__version__ != "1.5.2":
        pytest.skip(
            f"scikit-learn is {sklearn.__version__}, not the pinned 1.5.2. "
            "The tests above are what decide whether that is safe."
        )
