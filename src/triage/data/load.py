"""Load BAF from data/raw, verify checksums, write snake_case parquet to data/interim.

Rule 3: nothing this module writes ever goes into git. ``data/checksums.sha256``
is the one exception, and it holds hashes, not rows.

The CSV to parquet conversion runs in DuckDB rather than pandas: each variant is
a million rows and about 210 MB of CSV, and DuckDB streams it instead of holding
the whole frame in memory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import hydra
import pandas as pd
from omegaconf import DictConfig

from triage.config import file_checksum
from triage.runtime import CONFIG_PATH, start

log = logging.getLogger("triage")

CHECKSUM_FILE = "checksums.sha256"


def _sql_literal(path: Path) -> str:
    """Escape a path for inlining into SQL (DuckDB rejects a parameterised COPY target)."""
    return str(path).replace("'", "''")


def _column_selection(connection: duckdb.DuckDBPyConnection, source: Path, variant: str) -> str:
    """Keep exactly the contracted columns, so every variant lands the same shape.

    Variant V ships two extra columns, ``x1`` and ``x2``, that Base and Variant IV
    do not have. The champion is trained on Base, so the stress test has to score
    the same 32 columns; dropping them here -- loudly -- is what makes the three
    interim files comparable. A *missing* contracted column is an error.
    """
    from triage.data.contract import CONTRACT

    connection.execute(
        f"SELECT * FROM read_csv_auto('{_sql_literal(source)}', header = true) LIMIT 0"
    )
    header = [str(column[0]) for column in connection.description or []]

    missing = [column for column in CONTRACT if column not in header]
    if missing:
        raise ValueError(f"{source.name} is missing contracted columns: {missing}")

    extra = [column for column in header if column not in CONTRACT]
    if extra:
        log.warning("%s: dropping %d uncontracted column(s): %s", variant, len(extra), extra)

    return ", ".join(f'"{column}"' for column in CONTRACT)


def normalise_name(raw_name: str) -> str:
    """``"Variant IV.csv"`` -> ``"variant_iv"``."""
    return raw_name.removesuffix(".csv").strip().lower().replace(" ", "_")


def raw_path(cfg: DictConfig, variant: str) -> Path:
    """Where the published CSV for one variant lives."""
    return Path(cfg.paths.raw) / str(cfg.data.variants[variant])


def interim_path(cfg: DictConfig, variant: str) -> Path:
    """Where the normalised parquet for one variant lives."""
    return Path(cfg.paths.interim) / f"{variant}.parquet"


def _expected_files(cfg: DictConfig) -> dict[str, Path]:
    return {variant: raw_path(cfg, variant) for variant in cfg.data.variants}


def write_checksums(raw_dir: Path, out_path: Path, files: list[Path]) -> dict[str, str]:
    """Write ``data/checksums.sha256`` on the first download, in ``sha256sum`` format."""
    digests = {path.name: file_checksum(path) for path in sorted(files)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in digests.items()), encoding="utf-8"
    )
    log.info("wrote %s for %d file(s)", out_path, len(digests))
    return digests


def read_checksums(checksum_path: Path) -> dict[str, str]:
    """Parse a ``sha256sum``-format file into ``{filename: digest}``."""
    recorded = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        recorded[name.strip()] = digest.strip()
    return recorded


def verify_checksums(raw_dir: Path, checksum_path: Path) -> dict[str, str]:
    """Verify raw files against ``data/checksums.sha256``; raise on any mismatch.

    A mismatch means the file on disk is not the file the results were produced
    from, so it stops the run rather than warning.
    """
    recorded = read_checksums(checksum_path)
    if not recorded:
        raise ValueError(f"{checksum_path} is empty")

    missing, changed = [], []
    for name, digest in recorded.items():
        path = raw_dir / name
        if not path.exists():
            missing.append(name)
        elif file_checksum(path) != digest:
            changed.append(name)

    if missing or changed:
        raise ValueError(
            f"checksum verification failed against {checksum_path}: "
            f"missing={missing or 'none'}, changed={changed or 'none'}"
        )

    log.info("verified %d file(s) against %s", len(recorded), checksum_path)
    return recorded


def to_interim(cfg: DictConfig) -> dict[str, Path]:
    """Convert each configured variant to ``data/interim/<name>.parquet``."""
    interim_dir = Path(cfg.paths.interim)
    interim_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    connection = duckdb.connect()
    try:
        for variant in cfg.data.variants:
            source, target = raw_path(cfg, variant), interim_path(cfg, variant)
            if not source.exists():
                raise FileNotFoundError(
                    f"{source} is missing. Download it first (see data/README.md)."
                )
            selection = _column_selection(connection, source, variant)
            # DuckDB will not take a bound parameter for a COPY target, so the
            # paths are inlined. They come from the config, not from user input,
            # and single quotes are escaped.
            connection.execute(
                f"COPY (SELECT {selection} FROM read_csv_auto('{_sql_literal(source)}', "
                f"header = true)) TO '{_sql_literal(target)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            counted = connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [str(target)]
            ).fetchone()
            rows = int(counted[0]) if counted else 0
            log.info("%s -> %s (%s rows)", source.name, target.name, f"{rows:,}")
            written[variant] = target
    finally:
        connection.close()

    return written


def load_interim(cfg: DictConfig, variant: str = "base") -> pd.DataFrame:
    """Read one interim variant, honouring ``data.sample_frac`` for development runs.

    Sampling is seeded and stratified by month, so every month stays represented.
    Never report a result from a sampled run.
    """
    path = interim_path(cfg, variant)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the download and conversion first")

    frame = pd.read_parquet(path)

    fraction = float(cfg.data.sample_frac)
    if fraction < 1.0:
        time_column = str(cfg.data.time_column)
        frame = (
            frame.groupby(time_column, group_keys=False)
            .sample(frac=fraction, random_state=int(cfg.seed))
            .sort_index()
        )
        log.warning("sampled %s of %s to %d rows: development only", fraction, variant, len(frame))

    return frame.reset_index(drop=True)


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Verify the raw files, write checksums on first run, and convert to parquet."""
    start(cfg, "load")

    raw_dir = Path(cfg.paths.raw)
    checksum_path = Path(raw_dir).parent / CHECKSUM_FILE
    expected = _expected_files(cfg)

    missing = [str(path) for path in expected.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"raw files missing: {missing}. See data/README.md.")

    if checksum_path.exists():
        verify_checksums(raw_dir, checksum_path)
    else:
        write_checksums(raw_dir, checksum_path, list(expected.values()))

    for variant, path in to_interim(cfg).items():
        log.info("interim ready: %s -> %s", variant, path)


if __name__ == "__main__":
    main()
