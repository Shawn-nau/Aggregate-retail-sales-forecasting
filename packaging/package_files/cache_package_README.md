# Cache package — preprocessed M5 experiment samples

Supplementary package for the reproducibility kit of "Aggregate retail sales forecasting with child-level set representation learning". It contains the preprocessed sample caches so the reproduction run can **skip cache building entirely** (the slowest non-GPU step).

Assembly date: 2026-10-03. Cache version: `v5_child_targets_bottomup_global_boosting`. All caches were built with `--store-sku-sample-seed 42`.

## Contents

| Directory | Size | SKU fraction | Modes included | Enables |
|---|---|---|---|---|
| `100pct/` | 25 GB | 1.00 | holdout only | MAIN holdout results (paper Tables 1–2, Figures 2–3) |
| `100pct_rolling/` | 80 GB | 1.00 | holdout + rolling folds 1–5 | MAIN rolling results (paper Tables 3–4, Figures 4–5) |
| `10pct/` | 2.5 GB | 0.10 | holdout only | 10% robustness run |
| `30pct/` | 7.4 GB | 0.30 | holdout only | 30% robustness run |
| `50pct/` | 13 GB | 0.50 | holdout only | 50% robustness run |

Each directory contains `base_panel.pkl` plus `task__state_dept/`, `task__store_cat/`, `task__store_dept/` with the per-task sample pickles (`final_train_samples.pkl`, `internal_valid_samples.pkl`, `test_samples.pkl`, and — in `100pct_rolling/` — `rolling_fold_1..5_{train,valid}_samples.pkl`).

## How each directory was generated

`warm_m5_cache.py` in the main package's `code/scripts/`, run with `--cache-dir` set to that directory:

```bash
# 100pct (holdout)
python scripts/warm_m5_cache.py --data-dir data/m5 --cache-dir cache/100pct \
  --tasks store_dept,store_cat,state_dept --holdout \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac 1.0 --store-sku-sample-seed 42

# 100pct_rolling (holdout + rolling)
python scripts/warm_m5_cache.py --data-dir data/m5 --cache-dir cache/100pct_rolling \
  --tasks store_dept,store_cat,state_dept --holdout --rolling \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac 1.0 --store-sku-sample-seed 42
```

The 10/30/50pct directories use the same holdout command with `--store-sku-sample-frac 0.10|0.30|0.50`.

## How to use

1. Copy the needed directory next to the main package's code (or anywhere on a fast disk), e.g. `cp -r cache_package/100pct reproducibility_package/code/cache/   # run from the folder where you extracted both packages`.
2. Pass it via `--cache-dir` with the matching `--store-sku-sample-frac` and `--store-sku-sample-seed 42` in every experiment command (see the main package README, Flow A).
3. Do **not** rename files inside the directories, and do **not** mix directories with different SKU fractions — each experiment must use the cache built for its exact fraction.

## Transferring to a GPU server

These are large binary files; use rsync over SSH (resumable):

```bash
rsync -avP cache_package/ user@SERVER:~/m5/cache/
```

## Notes

- No cache is included for the 10% rolling robustness run — rebuild it with `--holdout --rolling --store-sku-sample-frac 0.10` (see main README, Flow B).
- The cache is regenerable from the raw M5 data in the main package (`data/m5/`) with the commands above; keeping the shipped copy avoids ~2 hours (holdout) to several hours (rolling) of preprocessing per fraction.
