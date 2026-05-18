# Cross-Level Aggregate Retail Forecasting with Set Representation Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of *"Aggregate retail sales forecasting with SKU-level set representation learning"* (Ma & Wang, 2026).

## Overview

This repository provides the complete experimental pipeline for cross-level retail forecasting on the M5 competition dataset. The framework encodes SKU-level future covariates into latent representations and aggregates them through learned set-based modules—**Gated Pooling**, **DeepSets**, and **Set Transformer**—to produce aggregate-level point and probabilistic forecasts.

**Key result**: Set-based cross-level models consistently outperform direct aggregate-level baselines, bottom-up approaches, and handcrafted cross-level aggregation strategies. The gains are most pronounced at more aggregated target levels, where the mismatch between target granularity and information granularity is greatest. Results are stable under rolling-origin evaluation and across SKU sample fractions from 10% to 100%.

## Repository structure

```
├── scripts/                     # All experiment scripts (Python + shell)
│   ├── prepare_m5_experiments.py      # Data pipeline and sample construction
│   ├── run_paper_experiment_plan.py   # End-to-end experiment orchestrator
│   ├── tune_m5_proposed_models.py     # Hyperparameter tuning (Optuna)
│   ├── run_m5_proposed_models.py      # Neural model training/evaluation
│   ├── m5_benchmarks.py               # sklearn benchmark implementations
│   ├── run_m5_full_benchmark.py       # Benchmark runner
│   ├── run_m5_ablation.py             # Ablation study runner
│   ├── run_m5_stat_tests.py           # Friedman/Nemenyi statistical tests
│   ├── make_m5_paper_tables.py        # LaTeX paper table generation
│   ├── compute_m5_metrics.py          # WRMSSE / WSPL metric computation
│   ├── generate_cd_diagrams_100pct.py # CD diagram generation
│   ├── build_robustness_tables.py     # Robustness summary table builder
│   ├── build_rolling_fold_tables.py   # Per-fold rolling table builder
│   ├── warm_m5_cache.py               # Cache pre-warming
│   ├── check_experiment_consistency.py # Post-run consistency checker
│   └── run_*.sh                       # Shell scripts for batch runs
├── results/work/
│   ├── v-3.tex                   # Paper source (LaTeX)
│   ├── references.bib            # Bibliography
│   ├── figures/                  # CD diagrams and other figures
│   └── {10,30,50,100}pct*/       # Experiment outputs per sample fraction
│       ├── 05_stat_tests/        # Statistical test outputs (CSV + PNG)
│       └── 06_paper_tables/      # Final LaTeX tables and summary CSVs
├── best_params_all_models.json   # Tuned hyperparameters for all models
├── CLAUDE.md                     # Detailed command reference for Claude Code
└── README.md                     # This file
```

## Quick start

### 1. Environment

```bash
# Python 3.10+ required
pip install numpy pandas scipy scikit-learn optuna lightgbm matplotlib torch

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

### 2. Data

Download the M5 competition dataset from [Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy/data) and place the CSV files in `data/m5/`:

```
data/m5/
├── calendar.csv
├── sales_train_evaluation.csv
├── sales_train_validation.csv
└── sell_prices.csv
```

### 3. Sanity check (10% sample, ~2 minutes)

```bash
export DATA_DIR=./data
export WORK_DIR=./results/work/sanity_debug
export CACHE_DIR=./cache/sanity

python scripts/run_paper_experiment_plan.py \
  --data-dir "$DATA_DIR" --work-dir "$WORK_DIR" --cache-dir "$CACHE_DIR" \
  --mode holdout --phases cache,sanity --tasks store_dept --sanity-task store_dept \
  --t-hist 56 --quantiles "" --store-sku-sample-frac 0.10
```

## Full reproduction pipeline

The complete experiment involves five phases. Commands below assume a Linux server with GPU and sufficient disk space. All paths use environment variables; adjust `$DATA_DIR`, `$WORK_DIR`, `$CACHE_DIR` to your setup.

### Phase 1: Cache warm-up (all sample fractions)

Pre-build the preprocessing cache for all SKU sample fractions used in the paper.

```bash
for FRAC in 0.10 0.30 0.50 1.00; do
  python scripts/warm_m5_cache.py \
    --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
    --tasks store_dept,store_cat,state_dept --holdout --rolling \
    --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
    --store-sku-sample-frac "$FRAC" --store-sku-sample-seed 42
done
```

### Phase 2: Hyperparameter tuning (per sample fraction, per model)

Tuning uses Optuna with two-stage search (broad exploration → top-k refinement). Below is the 100% sample workflow; adapt `--store-sku-sample-frac` and `--search-profile` for other fractions.

```bash
FRAC=1.00; PROFILE=full100; N_TRIALS=12; K=4
for MODEL in M3_FullSkuTemporalCNN DeepSets SetTransformer; do
  python scripts/tune_m5_proposed_models.py \
    --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR/tuning/${FRAC//.}pct/$MODEL" \
    --mode holdout --tasks store_dept,store_cat,state_dept \
    --model-name "$MODEL" --search-profile "$PROFILE" \
    --warm-start-params best_params_all_models.json \
    --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
    --store-sku-sample-frac "$FRAC" --store-sku-sample-seed 42 \
    --stage1-n-trials "$N_TRIALS" --stage1-train-frac 0.30 --stage1-epochs 8 --stage1-patience 3 \
    --stage2-top-k "$K" --stage2-train-frac 0.80 --stage2-epochs 16 --stage2-patience 5 \
    --final-epochs 32 --final-patience 8 \
    --storage "sqlite:///$WORK_DIR/optuna_${MODEL}_${FRAC//.}pct.db"
done
```

Tuning parameters for each sample fraction:

| Fraction | `--search-profile` | `--stage1-n-trials` | `--stage2-top-k` | `--stage1-epochs` | `--stage2-epochs` |
|----------|-------------------|---------------------|-------------------|--------------------|--------------------|
| 0.10     | auto → sample10   | 96                  | 8                 | 8                  | 16                 |
| 0.30     | auto → large30    | 40                  | 8                 | 10                 | 20                 |
| 0.50     | auto → large50    | 20                  | 6                 | 10                 | 20                 |
| 1.00     | full100           | 12                  | 4                 | 8                  | 16                 |

### Phase 3: Ablation, benchmark, and proposed model evaluation

Run the orchestrator for the remaining phases (skip cache and tune):

```bash
for FRAC in 0.10 0.30 0.50 1.00; do
  python scripts/run_paper_experiment_plan.py \
    --data-dir "$DATA_DIR" --work-dir "$WORK_DIR/${FRAC//.}pct" --cache-dir "$CACHE_DIR" \
    --mode holdout --phases ablation,benchmark,stats,tables \
    --tasks store_dept,store_cat,state_dept \
    --proposed-models M3_FullSkuTemporalCNN,DeepSets,SetTransformer \
    --ablation-params-model M3_FullSkuTemporalCNN \
    --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
    --store-sku-sample-frac "$FRAC" --store-sku-sample-seed 42 \
    --stage1-n-trials 96 --stage1-epochs 8 --stage2-top-k 8 \
    --stage2-epochs 16 --final-epochs 24 --final-patience 6 --stats-metric wspl
done
```

### Phase 4: Rolling-origin evaluation (100% sample)

```bash
python scripts/run_paper_experiment_plan.py \
  --data-dir "$DATA_DIR" --work-dir "$WORK_DIR/100pct_rolling" --cache-dir "$CACHE_DIR" \
  --mode rolling --phases ablation,benchmark,stats,tables \
  --tasks store_dept,store_cat,state_dept \
  --proposed-models M3_FullSkuTemporalCNN,DeepSets,SetTransformer \
  --ablation-params-model M3_FullSkuTemporalCNN \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac 1.00 --store-sku-sample-seed 42 \
  --stage1-n-trials 12 --stage1-epochs 8 --stage2-top-k 4 \
  --stage2-epochs 16 --final-epochs 32 --final-patience 8 --stats-metric wspl
```

### Phase 5: Generate CD diagrams and robustness tables

```bash
python scripts/generate_cd_diagrams_100pct.py
python scripts/build_robustness_tables.py
python scripts/build_rolling_fold_tables.py
```

### Phase 6: Consistency check

```bash
python scripts/check_experiment_consistency.py \
  --work-dir "$WORK_DIR/100pct" --mode holdout --metric wspl
```

## Models

### Proposed models (neural, GPU)
| Code name | Paper name | Description |
|-----------|-----------|-------------|
| `M3_FullSkuTemporalCNN` | Gated Pooling | Horizon-wise gated attention over SKU embeddings |
| `DeepSets` | DeepSets | Permutation-invariant distributional set encoder |
| `SetTransformer` | Set Transformer | Induced-set attention with cross-SKU interaction |

### Benchmark models (sklearn, CPU)
| Code name | Paper name | Type |
|-----------|-----------|------|
| `SeasonalNaive` | Seasonal Naive | Direct aggregate |
| `AggregateElasticNet` | Aggregate Elastic Net | Direct aggregate |
| `AggregateHistGB` | Aggregate HistGB | Direct aggregate |
| `BottomUpGlobalHistGB` | Bottom-up Global HistGB | Bottom-up |
| `ChildSummaryElasticNet` | Child-summary Elastic Net | Handcrafted cross-level |
| `ChildSummaryHistGB` | Child-summary HistGB | Handcrafted cross-level |

### Ablation models (neural, GPU)
| Code name | Paper name | Description |
|-----------|-----------|-------------|
| `M0_AggHistOnly` | AggHistOnly | Aggregate history only, no child features |
| `M1_AggHistFutureSummary` | AggHist Child-summary | Aggregate history + summarized child features |

## Evaluation metrics

- **WRMSSE** (Weighted Root Mean Squared Scaled Error): M5 point forecasting metric. Lower is better.
- **WSPL** (Weighted Scaled Pinball Loss): M5 probabilistic forecasting metric over 9 quantiles (0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995). Lower is better.
- **Friedman test + Nemenyi post-hoc**: Non-parametric statistical comparison of model ranks across 121 aggregate series. CD diagrams use α = 0.10.

## Key results (100% sample, holdout)

### Point forecasting (WRMSSE)
| Model | state×dept | store×cat | store×dept | Mean |
|-------|-----------|-----------|-----------|------|
| Set Transformer | 0.6358 | 0.6741 | 0.7594 | 0.6897 |
| Gated Pooling | 0.6444 | 0.6941 | 0.7429 | 0.6938 |
| DeepSets | 0.7057 | 0.6974 | 0.7730 | 0.7254 |

### Probabilistic forecasting (WSPL)
| Model | state×dept | store×cat | store×dept | Mean |
|-------|-----------|-----------|-----------|------|
| Set Transformer | 0.1686 | 0.1907 | 0.1949 | 0.1847 |
| DeepSets | 0.1761 | 0.1869 | 0.2022 | 0.1884 |
| Gated Pooling | 0.1858 | 0.1857 | 0.1944 | 0.1886 |

All detailed per-task results, robustness tables, and CD diagrams are available in `results/work/`.

## Hardware

Neural models were trained on an NVIDIA RTX 4090 (24 GB). Per-epoch training times:

| Model | store×dept | store×cat | state×dept | Avg. |
|-------|-----------|-----------|-----------|------|
| Gated Pooling | 14.8s | 14.7s | 16.3s | 15.3s |
| DeepSets | 25.5s | 20.7s | 29.0s | 25.1s |
| Set Transformer | 38.5s | 38.7s | 46.8s | 41.4s |

Estimated total GPU-hours for full reproduction: ~200–300 hours (including hyperparameter tuning across all sample fractions).

## Citation

```bibtex
@article{ma2026aggregate,
  title={Aggregate retail sales forecasting with SKU-level set representation learning},
  author={Ma, Shaohui and Wang, Shengkai},
  journal={Working paper},
  year={2026},
  institution={Nanjing Audit University, School of Business}
}
```

## License

MIT License. See `LICENSE` file for details.
