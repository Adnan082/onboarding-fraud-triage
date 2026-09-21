"""The cost frame and the generated validation report.

Both are pure functions over artefacts, so they are cheap to test and there is no
excuse for either being wrong: the cost frame decides which policy looks best, and
the report is what a model-risk reviewer reads.
"""

from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from triage.evaluation.validation import collect_findings, estimated_pages, render
from triage.policy.cost import cost_breakdown, cost_per_10k, sensitivity_table


@pytest.fixture(scope="module")
def cfg(repo_root):
    """The composed project config."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name="config")


# Ten applications: one missed fraud, one caught fraud sent to verify, two genuine
# sent to verify, three genuine to review, three approved.
DECISIONS = np.array(
    [
        "approve",
        "verify",
        "verify",
        "verify",
        "review",
        "review",
        "review",
        "approve",
        "approve",
        "approve",
    ]
)
LABELS = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
LOSS = np.array([1000.0, 2000.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0])


def test_cost_components_by_hand(cfg) -> None:
    """Each component counted explicitly, then scaled."""
    costs = cfg.policy.costs
    breakdown = cost_breakdown(DECISIONS, LABELS, LOSS, cfg)
    scale = float(costs.per_n_applications) / len(DECISIONS)

    assert breakdown["review"] == pytest.approx(float(costs.review_cost) * 3 * scale)
    assert breakdown["verify"] == pytest.approx(float(costs.verify_cost) * 3 * scale)
    # Only the approved fraud is a loss: the one sent to verify was caught.
    assert breakdown["missed_fraud"] == pytest.approx(
        float(costs.loss_proxy_multiplier) * 1000.0 * scale
    )
    # Two genuine applicants were sent to verify; the third verify was the fraud.
    assert breakdown["friction"] == pytest.approx(float(costs.friction_cost) * 2 * scale)
    assert breakdown["total"] == pytest.approx(
        sum(breakdown[k] for k in ("review", "verify", "missed_fraud", "friction"))
    )


def test_caught_fraud_costs_nothing_in_losses(cfg) -> None:
    """A fraud that was stopped is not a loss, whatever band stopped it."""
    all_verified = np.full(len(LABELS), "verify")
    assert cost_breakdown(all_verified, LABELS, LOSS, cfg)["missed_fraud"] == 0.0


def test_approving_every_fraud_is_the_expensive_failure(cfg) -> None:
    """The whole point of the frame: waving fraud through dominates the cost."""
    approve_all = np.full(len(LABELS), "approve")
    review_all = np.full(len(LABELS), "review")
    assert cost_per_10k(approve_all, LABELS, LOSS, cfg) > cost_per_10k(
        review_all, LABELS, LOSS, cfg
    )


def test_cost_is_scaled_so_policies_are_comparable(cfg) -> None:
    """Doubling the volume with the same mix must not change the cost per 10,000."""
    doubled = cost_per_10k(
        np.concatenate([DECISIONS, DECISIONS]),
        np.concatenate([LABELS, LABELS]),
        np.concatenate([LOSS, LOSS]),
        cfg,
    )
    assert doubled == pytest.approx(cost_per_10k(DECISIONS, LABELS, LOSS, cfg))


def test_sensitivity_table_covers_the_configured_ratios(cfg) -> None:
    """The ranking of two policies can flip with the assumption, so it is published."""
    table = sensitivity_table(DECISIONS, LABELS, LOSS, cfg)

    assert list(table["loss_ratio"]) == [float(r) for r in cfg.policy.costs.sensitivity_ratios]
    assert table["total"].is_monotonic_increasing, "a costlier missed fraud cannot cost less"
    assert table["review"].nunique() == 1, "only the loss term moves with the ratio"


def test_cost_rejects_mismatched_inputs(cfg) -> None:
    with pytest.raises(ValueError, match="same length"):
        cost_breakdown(DECISIONS[:-1], LABELS, LOSS, cfg)
    with pytest.raises(ValueError, match="no applications"):
        cost_breakdown(np.array([]), np.array([]), np.array([]), cfg)


# --- the validation report ---------------------------------------------------------


def _metrics_with_breach() -> dict:
    """The shape of metrics.json, with a policy that broke its promise."""
    return {
        "policy": {
            "chosen": {"alpha_fraud": 0.45, "alpha_legit": 0.01},
            "thresholds": {"n_fraud": 470},
            "months": {
                "6": {
                    "fraud_coverage": 0.59,
                    "genuine_exclusion_rate": 0.0158,
                    "by_age_band": [
                        {
                            "band": "20-29",
                            "n_fraud": 153,
                            "fraud_coverage": 0.458,
                            "review_share": 0.029,
                        },
                        {
                            "band": "60-69",
                            "n_fraud": 146,
                            "fraud_coverage": 0.712,
                            "review_share": 0.109,
                        },
                    ],
                }
            },
        }
    }


def test_a_breached_guarantee_is_a_high_finding() -> None:
    """The report must not bury the thing a reviewer most needs to see."""
    findings = collect_findings(_metrics_with_breach())
    titles = {f["title"]: f for f in findings}

    breach = next(f for t, f in titles.items() if "did not hold" in t)
    assert breach["rating"] == "High"
    assert "exchangeab" in breach["body"].lower()
    assert breach["recommendation"]


def test_loose_coverage_is_raised_with_its_cause() -> None:
    """Over-delivery is still a finding, and the small calibration set is why."""
    findings = collect_findings(_metrics_with_breach())
    coverage = next(f for f in findings if "looser than intended" in f["title"])

    assert coverage["rating"] == "Medium"
    assert "470" in coverage["body"]


def test_an_uneven_burden_is_raised_and_not_fixed_with_age() -> None:
    """The recommendation must not propose using age at decision time."""
    findings = collect_findings(_metrics_with_breach())
    burden = next(f for f in findings if "unevenly" in f["title"])

    assert burden["rating"] == "High"
    assert "legal sign-off" in burden["recommendation"]
    assert "do not correct this with age-specific" in burden["recommendation"].lower()


def test_an_empty_artefact_produces_an_honest_report() -> None:
    """No results must read as no results, never as a clean bill of health."""
    findings = collect_findings({})
    assert len(findings) == 1
    assert "not been run" in findings[0]["body"]

    report = render({})
    assert "Not generated" in report
    assert "# Model validation report" in report


def test_the_report_states_what_it_cannot_show() -> None:
    """Section 14's scoping rules are not optional."""
    report = render(_metrics_with_breach())

    assert "synthetic" in report
    assert "not UK data" in report
    assert "no vulnerability analysis is possible" in report
    assert "No saving is claimed" in report


def test_the_generated_report_stays_inside_its_page_budget(repo_root) -> None:
    """Section 14 caps the validation report at two pages when exported.

    Tested rather than trusted, because a validation report grows by accretion --
    every run adds something worth saying -- and the thing the cap protects is a
    reviewer's attention, which does not grow.
    """
    report = (repo_root / "reports" / "validation_report.md").read_text(encoding="utf-8")
    pages = estimated_pages(report)
    assert pages <= 2.0, (
        f"reports/validation_report.md renders to about {pages:.2f} pages, over the "
        "two-page cap. Trim the prose in triage.evaluation.validation.render, not "
        "the findings."
    )


def test_the_report_names_its_provenance() -> None:
    """A reviewer must be able to tell which run produced it."""
    metrics = _metrics_with_breach()
    metrics["policy"]["context"] = {"config_hash": "abc12345", "git_sha": "deadbeef"}

    report = render(metrics)
    assert "abc12345" in report
    assert "deadbeef" in report
