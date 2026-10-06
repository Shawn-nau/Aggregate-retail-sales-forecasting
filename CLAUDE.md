# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

M5 cross-level retail forecasting experiments — point and probabilistic (quantile) aggregate forecasting benchmarked on the M5 competition dataset. The project compares classical/sklearn benchmarks against PyTorch neural models (DeepSets, SetTransformer, SkuTemporalCNN) using WRMSSE (point) and WSPL (quantile) metrics, with Friedman/Nemenyi statistical tests and LaTeX paper table generation.

## Environment and common commands

Set environment variables before running:

```bash
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

**Fast sanity check** (tests data loading, GPU, model execution, output writing):

```bash
python scripts/run_paper_experiment_plan.py \
  --data-dir "$DATA_DIR" --work-dir "$WORK_DIR/sanity_debug" --cache-dir "$CACHE_DIR" \
  --mode holdout --phases cache,sanity --tasks store_dept --sanity-task store_dept \
  --t-hist 56 --quantiles "" --store-sku-sample-frac 0.10
```

**End-to-end holdout paper run** (the primary workflow):

```bash
python scripts/run_paper_experiment_plan.py \
  --data-dir "$DATA_DIR" --work-dir "$WORK_DIR/holdout" --cache-dir "$CACHE_DIR" \
  --mode holdout --tasks store_dept,store_cat,state_dept \
  --proposed-models M3_FullSkuTemporalCNN,DeepSets,SetTransformer \
  --ablation-params-model M3_FullSkuTemporalCNN --t-hist 56 --horizon 28 \
  --valid-size 4 --test-size 4 --internal-valid-size 3 --store-sku-sample-frac 0.10 \
  --store-sku-sample-seed 42 --stage1-n-trials 96 --stage1-epochs 8 --stage2-top-k 8 \
  --stage2-epochs 16 --final-epochs 24 --final-patience 6 --stats-metric wspl
```

**Post-run consistency check**:

```bash
python scripts/check_experiment_consistency.py --work-dir "$WORK_DIR/holdout" --mode holdout --metric wspl
```

**Warm cache for larger samples** (30%, 50%, 100%):

```bash
python scripts/warm_m5_cache.py --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
  --tasks store_dept,store_cat,state_dept --holdout \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac 0.30 --store-sku-sample-seed 42
```

**Force rebuild preprocessing cache** (required if old cache lacks `y_child_target`):

```bash
python scripts/warm_m5_cache.py --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
  --tasks store_dept,store_cat,state_dept --holdout --rolling --force-rebuild-cache
```

## Architecture

### Data pipeline (`prepare_m5_experiments.py`)

The central data module. Defines `M5ExperimentConfig` (dataclass with all experiment hyperparameters) and builds the entire sample cache.

1. `read_m5_raw()` loads M5 CSVs (sales_train_evaluation.csv or sales_train_validation.csv, calendar.csv, sell_prices.csv)
2. `build_m5_item_store_panel()` melts the wide-format sales table into long format, merges calendar features (wday, month, year, snap flags, event flags) and price features (sell_price, price_change_ratio), creates `sku_id = item_id + "__" + store_id`, and optionally samples store-SKU units via `store_sku_sample_frac`
3. `add_task_aggregate_id()` creates aggregate labels based on task type (store_dept → store_id__dept_id, etc.)
4. `_build_task_array_store()` creates dense numpy arrays per aggregate unit (y_full, feature matrices, agg_y_full) for efficient slice-based sample generation
5. Origin index splitting: `make_holdout_origin_split()` produces train_origins_for_valid, valid_origins, trainval_origins_for_test, test_origins. `make_strict_internal_valid_split()` carves internal validation from trainval. `make_expanding_window_folds()` produces rolling folds
6. `build_m5_experiment_samples()` is the top-level function that builds/caches all task packages and split samples (outer_train/valid, final_train, internal_valid, test, rolling folds). Called by every runner script
7. `load_task_split_samples()` is the lazy-loading entry point used by neural model scripts

Cache version is `"v5_child_targets_bottomup_global_boosting"` — samples include `y_child_target` (child-level future demand) required by `BottomUpGlobalHistGB`.

### Experiment orchestration (`run_paper_experiment_plan.py`)

The recommended end-to-end runner. Phase order: cache → sanity → tune → ablation → benchmark → stats → tables. Each phase calls a dedicated script via subprocess. Parameters are split into two categories:
- `split_control_args`: must be identical across all phases (t_hist, horizon, valid_size, test_size, etc.)
- `runtime_cache_args`: cache directory and progress flags

Tuning always uses holdout-mode internal validation regardless of the final evaluation mode. Benchmark/ablation/stats/tables follow the selected `--mode`.

### Benchmark models (`m5_benchmarks.py`)

sklearn-based benchmarks, all inheriting from `BenchmarkBase`:
- **Aggregate-level**: `SeasonalNaiveAggregateBenchmark`, `AggregateElasticNet`, `AggregateHistGB` — predict directly from aggregate history features + mean future covariates
- **Cross-level handcrafted**: `ChildSummaryElasticNet`, `ChildSummaryHistGB` — use rich child-summary features (distribution stats, top-k shares, HHI) + future covariate summaries
- **Bottom-up global**: `BottomUpGlobalBoostingBenchmark` — trains a single multi-output HistGB at the child store-SKU level pooled across all aggregates, then sums child predictions to aggregate forecasts. Requires `y_child_target` in samples

Benchmark suite factory: `build_benchmark_suite()`. Runner: `run_benchmark_suite_on_task()` supports both `holdout` and `rolling` modes, returns point and quantile results with optional series-level rows.

### Proposed neural models (`run_m5_proposed_models.py`)

PyTorch-based models with configurable architectures. The file is large — core components:
- **Scalers**: `FeatureStandardScaler`, `ArrayStandardScaler` for feature normalization
- **Datasets/DataLoaders**: batching of variable-sized aggregate samples
- **Model architectures**: configurable via `--model-names` — `M3_FullSkuTemporalCNN` (SkuTemporalCNN), `DeepSets`, `SetTransformer`
- **Training**: `FitConfig` controls epochs, patience, batch_size, hidden_dim, etc. `TrainRegularizationConfig` controls regularization
- **Evaluation**: `run_one_setting()` fits and evaluates a model on a task, producing both block-level and series-level metrics
- Model name aliases are resolved via `CANONICAL_MODEL_MAP`

### Hyperparameter tuning (`tune_m5_proposed_models.py`)

Two-stage Optuna tuning with memory-aware search profiles and OOM resilience.

**Search profiles** (auto-mapped from `--store-sku-sample-frac` via `--search-profile auto`):

| Profile | SKU frac | Purpose |
|---|---|---|
| `sample10` | 0.10 | Initial/sanity tuning |
| `large30` | 0.30 | Main large-sample retuning |
| `large50` | 0.50 | Stability confirmation |
| `full100` | 1.00 | Final narrow confirmation |

Profiles narrow the hyperparameter ranges as the sample fraction increases — narrower LR, dropout, weight decay, hidden dim, CNN blocks and batch-size ranges. SetTransformer gets especially conservative batch sizes at 50% and 100%.

Pipeline:
1. **Stage 1**: broad exploration with many trials, short training, small train fraction. OOM trials are pruned (not retried).
2. **Stage 2**: top-k refinement with more epochs, larger train fraction. OOM trials are retried with progressively smaller batch sizes.
3. **Final**: best params from stage 2 get a full training run. OOM is also retried with smaller batches.

Outputs `best_params.json` per model, consumed by `run_m5_proposed_models.py` via `--params-json`.

**Warm-start**: `--warm-start-params` accepts either a single-model `best_params.json` or an all-models JSON (`best_params_all_models.json`). When an all-models JSON is given, the tuner extracts the parameters matching `--model-name` and enqueues them as the first Optuna trial.

**Storage**: `--storage "sqlite:///path/to/optuna.db"` persists Optuna study state for resumption.

Key new CLI args: `--search-profile`, `--warm-start-params`, `--no-enqueue-warm-start`, `--storage`, `--stage1-patience`, `--stage2-patience`, `--stage1-max-folds`, `--stage2-max-folds`, `--skip-final-stage`, `--stage1-task`, `--stage2-tasks`, `--pruner`, `--sampler`.

### Large-sample tuning workflow

For 30%, 50%, or 100% SKU samples, tuning must be run independently stage-by-stage (the orchestrator's `--phases tune` does not pass the new search-profile/warm-start flags). Run tuning per model per sample fraction:

```bash
# 30% tuning (main large-sample retuning)
for MODEL in M3_FullSkuTemporalCNN DeepSets SetTransformer; do
  python scripts/tune_m5_proposed_models.py \
    --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR/01_tuning_30pct/${MODEL}" \
    --mode holdout --tasks store_dept,store_cat,state_dept \
    --model-name "$MODEL" --search-profile auto \
    --warm-start-params best_params_all_models.json \
    --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
    --store-sku-sample-frac 0.30 --store-sku-sample-seed 42 \
    --stage1-n-trials 40 --stage1-train-frac 0.40 --stage1-epochs 10 --stage1-patience 4 \
    --stage2-top-k 8 --stage2-train-frac 1.00 --stage2-epochs 20 --stage2-patience 6 \
    --final-epochs 32 --final-patience 8 \
    --storage "sqlite:///$WORK_DIR/optuna_${MODEL}_30pct.db"
done

# 50% confirmation (fewer trials, warm-start from 30% result)
for MODEL in M3_FullSkuTemporalCNN DeepSets SetTransformer; do
  python scripts/tune_m5_proposed_models.py \
    --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR/01_tuning_50pct/${MODEL}" \
    --mode holdout --tasks store_dept,store_cat,state_dept \
    --model-name "$MODEL" --search-profile auto \
    --warm-start-params "$WORK_DIR/01_tuning_30pct/${MODEL}/best_params.json" \
    --t-hist 56 --store-sku-sample-frac 0.50 --store-sku-sample-seed 42 \
    --stage1-n-trials 20 --stage1-train-frac 0.40 --stage1-epochs 10 --stage1-patience 4 \
    --stage2-top-k 6 --stage2-train-frac 1.00 --stage2-epochs 20 --stage2-patience 6 \
    --final-epochs 32 --final-patience 8 \
    --storage "sqlite:///$WORK_DIR/optuna_${MODEL}_50pct.db"
done

# 100% final confirmation (narrow search, few trials)
for MODEL in M3_FullSkuTemporalCNN DeepSets SetTransformer; do
  python scripts/tune_m5_proposed_models.py \
    --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR/01_tuning_100pct/${MODEL}" \
    --mode holdout --tasks store_dept,store_cat,state_dept \
    --model-name "$MODEL" --search-profile full100 \
    --warm-start-params "$WORK_DIR/01_tuning_50pct/${MODEL}/best_params.json" \
    --t-hist 56 --store-sku-sample-frac 1.00 --store-sku-sample-seed 42 \
    --stage1-n-trials 12 --stage1-train-frac 0.30 --stage1-epochs 8 --stage1-patience 3 \
    --stage2-top-k 4 --stage2-train-frac 0.80 --stage2-epochs 16 --stage2-patience 5 \
    --final-epochs 32 --final-patience 8 \
    --storage "sqlite:///$WORK_DIR/optuna_${MODEL}_100pct.db"
done
```

After tuning, the benchmark/ablation/stats/tables phases (11.3-11.7 in README) can be run with the tuned params and matching `--store-sku-sample-frac`. The orchestrator (`run_paper_experiment_plan.py`) can still run these phases for any sample fraction, provided the cache was pre-warmed with the matching `--store-sku-sample-frac` and `--store-sku-sample-seed`.

### Ablation (`run_m5_ablation.py`)

Runs three ablation models against selected benchmark baselines:
- `M0_AggHistOnly`: aggregate history features only (no child-level info)
- `M1_AggHistFutureSummary`: aggregate history + future covariate summary (no child structure)
- `M3_FullSkuTemporalCNN`: the full proposed model

Uses the same neural training infrastructure as `run_m5_proposed_models.py`.

### Metrics (`compute_m5_metrics.py`)

Computes M5 competition metrics with proper scaling:
- `compute_wrmsse_from_samples()`: Weighted RMSSE using M5 scaling weights
- `compute_wspl_from_samples()`: Weighted Scaled Pinball Loss with quantile-level detail
- `build_task_aggregate_daily()`: creates task-level aggregate daily dataframe
- `prepare_aggregate_history_store()`: creates per-aggregate history store for scaling

### Statistical tests (`run_m5_stat_tests.py`)

Friedman test with post-hoc Nemenyi analysis:
- Merges benchmark + proposed + (filtered) ablation rows
- Supports two analysis levels:
  - **series-level** (default, preferred): forms `task × agg_id` blocks by averaging per-origin series metrics within each block, then ranks models
  - **block-level**: uses `task × fold` blocks
- Outputs: summary CSV, pairwise p-value matrix, CD diagram PNG, LaTeX table

### Paper tables (`make_m5_paper_tables.py`)

Generates LaTeX tables with bold/underline formatting:
- Table 2: main point forecasting (WRMSSE)
- Table 3: main quantile forecasting (WSPL)
- Table 4: ablation study
- Table 5A/5B: robustness (data scarcity, model capacity) — requires separate `run_m5_robustness.py` inputs

### Optional: experiment tracking (`experiment_tracking.py`)

W&B integration wrapper (`WandbRunWrapper`), Optuna import helper, JSON-safe serialization.

### Optional: robustness (`run_m5_robustness.py`)

Separate robustness experiments for data scarcity (varying training origins) and model capacity sensitivity.

## Key design rules

1. **Holdout vs rolling consistency**: Holdout always uses fold=0. Rolling uses fold=1,2,... Both must share identical `split_control_args`.
2. **Tuning always uses holdout**: Even when the final benchmark mode is rolling, tuning uses holdout-style internal validation. The cache must therefore be warmed with both `--rolling` and `--holdout` when running rolling experiments that include a tuning phase.
3. **Bottom-up baseline**: The old `BottomUpSeasonalNaive` is fully replaced by `BottomUpGlobalHistGB`. The consistency checker asserts the new is present and the old is absent.
4. **Ablation stat-test participants**: Only M0 and M1 participate in statistical comparisons; M3 is generated by the ablation run but filtered out by `--ablation-models`.
5. **Cache versioning**: Samples must include `y_child_target`. If old cache files are reused, rebuild with `--force-rebuild-cache`.
6. **Thread limiting**: sklearn benchmarks use `threadpool_limits(limits=1)` to prevent CPU oversubscription. Torch uses single-threaded CPU settings.
7. **Large-sample tuning requires standalone tuner calls**: The orchestrator (`run_paper_experiment_plan.py`) does not pass the new search-profile/warm-start/storage/patience flags. For 30%/50%/100% SKU fractions, run `tune_m5_proposed_models.py` directly with `--search-profile auto` and `--warm-start-params`, then run the benchmark/ablation/stats/tables phases via the orchestrator with `--phases ablation,benchmark,stats,tables` (skipping cache and tune).
8. **Search-profile auto-mapping**: `--search-profile auto` maps `--store-sku-sample-frac` to the correct profile: ≥0.95 → full100, ≥0.45 → large50, ≥0.25 → large30, else sample10.
9. **OOM handling**: Stage-1 OOM trials are pruned. Stage-2 and final-stage OOM are retried with progressively smaller batch sizes (8→4→2→1). SetTransformer uses narrower batch ranges than SkuTemporalCNN/DeepSets at 50%/100%.
10. **Warm-start JSON formats**: `--warm-start-params` accepts either a single-model `best_params.json` (direct dict) or an all-models JSON (`best_params_all_models.json`) — the tuner automatically extracts the matching model's params using `TUNED_PARAM_KEY_CANDIDATES`.
11. **Separate caches per sample fraction**: Each `--store-sku-sample-frac` value produces different SKU sampling. Use separate cache directories or ensure the cache key differentiates by seed+frac. The same `--store-sku-sample-seed` must be used across cache, tuning, ablation, and benchmark for a given experiment.
