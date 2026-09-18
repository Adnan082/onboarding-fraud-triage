"""Pandera data contract: types, ranges, category sets and nullability.

**Frozen on 2026-09-18** from the first verified load of BAF Base (1,000,000 rows,
months 0-7), cross-checked against Variant IV and Variant V. Any later change
needs an entry in docs/DECISIONS.md.

How the bounds were chosen:

- **Closed domains get exact sets.** Binaries, the categorical level sets, the
  nine age decades, months 0-7, and ``income`` on its decile grid 0.1-0.9. These
  are the checks that catch a broken upstream feed, including the ``income x 10``
  bug, which lands in 1.0-9.0 and is rejected.
- **Open-ended numerics get the observed Base range plus documented headroom.**
  Variant IV and Variant V drift a little past Base's exact minima and maxima
  (``velocity_6h`` reaches 16,802 against Base's 16,716, for instance), so an
  exact-range contract would reject the stress tests for no good reason. The
  headroom is generous enough to absorb that drift and far too tight to absorb a
  broken column.

Two findings from the first load, both recorded in docs/DECISIONS.md:

- ``credit_risk_score`` is genuinely negative for 1.44% of applications. It is a
  score, not a count, so **negative values there are real, not missing**, and -1
  must not be treated as a sentinel.
- ``velocity_6h`` is negative for 44 of a million rows (minimum -170.6), which is
  not physically meaningful and is not a documented sentinel. The contract allows
  it, because it is in the published data; the feature layer flags it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import pandas as pd
import pandera.pandas as pa
from omegaconf import DictConfig

from triage.runtime import CONFIG_PATH, start

log = logging.getLogger("triage")

CONTRACT_VERSION = "1.0"
FROZEN_ON = "2026-09-18"
FROZEN_FROM = "BAF Base, 1,000,000 rows, months 0-7"


@dataclass(frozen=True)
class ColumnSpec:
    """One column of the frozen contract."""

    kind: str  # "int", "float" or "category"
    lo: float | None = None
    hi: float | None = None
    levels: tuple[str | int, ...] = ()
    sentinel: str = ""  # how missing arrives, if it does
    note: str = ""
    observed: str = ""  # what the first verified load actually showed

    def checks(self) -> list[pa.Check]:
        """The pandera checks this spec implies."""
        if self.levels:
            return [pa.Check.isin(list(self.levels))]
        if self.lo is not None and self.hi is not None:
            return [pa.Check.in_range(self.lo, self.hi)]
        return []

    @property
    def allowed(self) -> str:
        """Human-readable domain, for reports/data_contract.md."""
        if self.levels:
            shown = ", ".join(str(level) for level in self.levels)
            return f"{{{shown}}}"
        return f"[{self.lo:g}, {self.hi:g}]" if self.lo is not None else "any"


BINARY = ColumnSpec("int", levels=(0, 1), note="binary flag")

CONTRACT: dict[str, ColumnSpec] = {
    # --- label and time -----------------------------------------------------------
    "fraud_bool": ColumnSpec("int", levels=(0, 1), note="label", observed="1.103% positive"),
    "month": ColumnSpec(
        "int",
        levels=tuple(range(8)),
        note="the only split key (rule 1)",
        observed="0-7, 96,843-150,936 rows each",
    ),
    # --- protected attribute: measurement only, never a model input (rule 4) -------
    "customer_age": ColumnSpec(
        "int",
        levels=(10, 20, 30, 40, 50, 60, 70, 80, 90),
        note="rounded to the decade; measurement and training-constraint use only",
        observed="18.3% are 50 or over",
    ),
    "income": ColumnSpec(
        "float",
        0.1,
        0.9,
        note="decile grid; the closed range is what rejects the income x 10 bug",
        observed="nine values, 0.1-0.9",
    ),
    # --- categoricals ---------------------------------------------------------------
    "payment_type": ColumnSpec(
        "category",
        levels=("AA", "AB", "AC", "AD", "AE"),
        observed="AB most frequent, AE only 289 rows",
    ),
    "employment_status": ColumnSpec(
        "category", levels=("CA", "CB", "CC", "CD", "CE", "CF", "CG"), observed="CA 73%, CB 14%"
    ),
    "housing_status": ColumnSpec(
        "category", levels=("BA", "BB", "BC", "BD", "BE", "BF", "BG"), observed="BC 37%, BB 26%"
    ),
    "source": ColumnSpec("category", levels=("INTERNET", "TELEAPP"), observed="INTERNET 99.3%"),
    "device_os": ColumnSpec(
        "category",
        levels=("windows", "macintosh", "linux", "x11", "other"),
        observed="other 34%, linux 33%, windows 26%",
    ),
    # --- binaries --------------------------------------------------------------------
    "email_is_free": BINARY,
    "phone_home_valid": BINARY,
    "phone_mobile_valid": BINARY,
    "has_other_cards": BINARY,
    "foreign_request": BINARY,
    "keep_alive_session": BINARY,
    # --- numerics --------------------------------------------------------------------
    "name_email_similarity": ColumnSpec("float", 0.0, 1.0, observed="0.0000014-0.999999"),
    "prev_address_months_count": ColumnSpec(
        "int", -1, 500, sentinel="-1", note="71.3% missing", observed="-1 to 383 (variant IV: 399)"
    ),
    "current_address_months_count": ColumnSpec(
        "int", -1, 500, sentinel="-1", note="0.43% missing", observed="-1 to 428"
    ),
    "days_since_request": ColumnSpec("float", 0.0, 120.0, observed="0-78.5"),
    "intended_balcon_amount": ColumnSpec(
        "float",
        -30.0,
        150.0,
        sentinel="any negative",
        note="74.3% negative; no value is exactly -1",
        observed="-15.53 to 112.96",
    ),
    "zip_count_4w": ColumnSpec("int", 0, 8000, observed="1-6,700 (variant V: 6,830)"),
    "velocity_6h": ColumnSpec(
        "float",
        -500.0,
        20000.0,
        note="44 negative rows in Base: a published data quirk, flagged in features",
        observed="-170.6 to 16,715.6",
    ),
    "velocity_24h": ColumnSpec("float", 1000.0, 11000.0, observed="1,300.3-9,506.9"),
    "velocity_4w": ColumnSpec("float", 2500.0, 8000.0, observed="2,825.7-6,994.8"),
    "bank_branch_count_8w": ColumnSpec("int", 0, 3000, observed="0-2,385"),
    "date_of_birth_distinct_emails_4w": ColumnSpec("int", 0, 60, observed="0-39"),
    "credit_risk_score": ColumnSpec(
        "int",
        -250,
        500,
        note="negative values are REAL scores, not missing: never flag -1 here",
        observed="-170 to 389; 1.44% negative",
    ),
    "bank_months_count": ColumnSpec(
        "int", -1, 40, sentinel="-1", note="25.4% missing", observed="-1 to 32"
    ),
    "proposed_credit_limit": ColumnSpec(
        "float",
        150.0,
        2500.0,
        note="twelve discrete tiers in Base; kept as a range so a new tier is not a breach",
        observed="190-2,100",
    ),
    "session_length_in_minutes": ColumnSpec(
        "float", -1.0, 120.0, sentinel="-1", note="0.20% missing", observed="-1 to 85.9"
    ),
    "device_distinct_emails_8w": ColumnSpec(
        "int", -1, 10, sentinel="-1", note="0.04% missing", observed="-1, 0, 1, 2 only"
    ),
    "device_fraud_count": ColumnSpec(
        "int", 0, 10, note="zero variance in Base: dropped from features", observed="always 0"
    ),
}

SENTINEL_COLUMNS: dict[str, str] = {
    name: spec.sentinel for name, spec in CONTRACT.items() if spec.sentinel
}
ZERO_VARIANCE_COLUMNS: tuple[str, ...] = ("device_fraud_count",)

_DTYPES: dict[str, type] = {"int": int, "float": float, "category": str}


@dataclass
class ValidationResult:
    """What one validation run found."""

    name: str
    rows: int
    passed: bool
    failures: list[str] = field(default_factory=list)


def build_schema(strict: bool = True) -> pa.DataFrameSchema:
    """Return the frozen BAF schema.

    ``strict`` rejects any column the contract does not know about, which is the
    batch equivalent of the API's ``extra="forbid"``.
    """
    columns = {
        name: pa.Column(
            _DTYPES[spec.kind],
            checks=spec.checks(),
            nullable=False,
            coerce=True,
            required=True,
            description=spec.note or None,
        )
        for name, spec in CONTRACT.items()
    }
    return pa.DataFrameSchema(
        columns,
        strict=strict,
        ordered=False,
        name=f"baf-contract-v{CONTRACT_VERSION}",
    )


def validate(frame: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    """Validate a frame against the contract. Raises ``pa.errors.SchemaErrors``."""
    return build_schema().validate(frame, lazy=lazy)


def check(frame: pd.DataFrame, name: str = "frame") -> ValidationResult:
    """Validate without raising: returns what failed, for reports and the monitor."""
    try:
        validate(frame)
    except pa.errors.SchemaErrors as errors:
        cases = errors.failure_cases
        failures = [
            f"{row.column}: {row.check} ({int(row.get('n', 1)) if hasattr(row, 'get') else 1})"
            for row in cases.drop_duplicates(subset=["column", "check"]).itertuples()
        ]
        return ValidationResult(name=name, rows=len(frame), passed=False, failures=failures)
    except pa.errors.SchemaError as error:
        return ValidationResult(name=name, rows=len(frame), passed=False, failures=[str(error)])
    return ValidationResult(name=name, rows=len(frame), passed=True)


def write_contract_doc(out_path: Path, results: list[ValidationResult] | None = None) -> None:
    """Write ``reports/data_contract.md`` from the schema itself."""
    lines = [
        "# Data contract",
        "",
        f"Version **{CONTRACT_VERSION}**, frozen **{FROZEN_ON}** from {FROZEN_FROM}.",
        "",
        "Generated by `make contract` from `src/triage/data/contract.py`. Do not edit by",
        "hand. Any change to this contract needs an entry in `docs/DECISIONS.md`.",
        "",
        "Missing values arrive as negative numbers, never as nulls: the data has no nulls",
        "at all. The `Missing` column below says how a missing value announces itself.",
        "",
        "| Column | Type | Allowed | Missing | Observed on the first load | Note |",
        "|---|---|---|---|---|---|",
    ]
    for name, spec in CONTRACT.items():
        lines.append(
            f"| `{name}` | {spec.kind} | {spec.allowed} | {spec.sentinel or '-'} "
            f"| {spec.observed or '-'} | {spec.note or '-'} |"
        )

    if results:
        lines += [
            "",
            "## Validation",
            "",
            "| Dataset | Rows | Result | Failures |",
            "|---|---|---|---|",
        ]
        for result in results:
            status = "pass" if result.passed else "FAIL"
            failures = "; ".join(result.failures) if result.failures else "-"
            lines.append(f"| {result.name} | {result.rows:,} | {status} | {failures} |")

    lines += [
        "",
        "## What this contract is for",
        "",
        "- It rejects a broken upstream feed before the model ever scores it. The",
        "  `income x 10` bug in the monitoring experiments is caught here, in batch and",
        "  at the API, because `income` is confined to its 0.1-0.9 decile grid.",
        "- Open-ended numeric bounds carry headroom over the observed Base range, so the",
        "  Variant IV and Variant V stress tests are not rejected for drifting slightly",
        "  past Base's exact minima and maxima.",
        "- `credit_risk_score` may be negative: those are real scores, not missing values.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", out_path)


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Validate every interim variant and write ``reports/data_contract.md``."""
    from triage.data.load import interim_path

    start(cfg, "contract")

    results = []
    for variant in cfg.data.variants:
        path = interim_path(cfg, variant)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run the loader first")
        result = check(pd.read_parquet(path), name=variant)
        log.info(
            "%s: %s (%s rows)%s",
            variant,
            "pass" if result.passed else "FAIL",
            f"{result.rows:,}",
            "" if result.passed else f" -> {result.failures}",
        )
        results.append(result)

    write_contract_doc(Path(cfg.paths.reports) / "data_contract.md", results)

    if not all(result.passed for result in results):
        raise ValueError("the contract rejected data it should accept: see the log above")


if __name__ == "__main__":
    main()
