# data/

Row-level data never goes in git (CLAUDE.md rule 3). Only this file and
`checksums.sha256` are committed.

## Getting the data

The dataset is the public **Bank Account Fraud (BAF)** suite from NeurIPS 2022:
<https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022>

You need Kaggle credentials in `~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME` and
`KAGGLE_KEY` in the environment. Then:

```bash
make data
```

That downloads **Base**, **Variant IV** and **Variant V** into `raw/`, verifies
them against `checksums.sha256` (writing it on the first run), and converts them
to `interim/{base,variant_iv,variant_v}.parquet` with snake_case names.

## Layout

| Path | Contents |
|---|---|
| `raw/` | Files exactly as Kaggle publishes them (names may contain spaces) |
| `interim/` | Normalised parquet, one file per variant |
| `checksums.sha256` | SHA-256 of each raw file, written on the first download |

## Licence

Check the licence on the Kaggle page before publishing anything derived from the
data. That check is the owner's to make (CLAUDE.md section 3).

The data is **synthetic**, generated from an unnamed bank's applications. It is
not UK data and it is not real customers.
