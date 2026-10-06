# Design: OLS-Reconciled Benchmark (`Reconcilation`)

Date: 2026-08-14

## Goal

Add a new point-forecasting benchmark model, `Reconcilation`, that combines two existing benchmarks with standard OLS-based hierarchical reconciliation:

- **Top-level (aggregate) base forecast**: produced by the existing `AggregateHistGB` benchmark.
- **Bottom-level (child store-SKU) base forecasts**: produced by the existing `BottomUpGlobalHistGB` benchmark.
- **Reconciliation**: standard OLS projection onto the coherent subspace (Hyndman et al.), applied per sample and per horizon step. Only the reconciled **aggregate-level** forecast is kept.

The model is **point-forecast only** (no quantiles). All existing experiment results stay untouched; the new benchmark is run only for the 5 experiment settings listed below, and its rows are then merged into the existing result CSVs, after which stats/tables are regenerated (full integration).

## Clarified decisions (from brainstorming)

1. **OLS formulation**: standard projection OLS — ŷ̃ = S(S′S)⁻¹S′ŷ on the two-level hierarchy [top; children], per horizon step h. Closed form per sample:
   `ŷ̃_top[h] = (N·ŷ_top[h] + Σᵢ ŷ_childᵢ[h]) / (N+1)`
   where N = number of child store-SKUs in the sample, ŷ_childᵢ are the clipped (≥0) child forecasts from `BottomUpGlobalHistGB`. Degenerate guard: if N == 0, ŷ̃_top = ŷ_top.
2. **Model name**: `Reconcilation` (exact spelling as chosen by user; appears in CSVs, stat tests, Table 2, and CD diagrams).
3. **Scope of experiments** (5 settings, all with seed 42):
   - holdout: frac 0.10 (`results/work/10pct`), 0.30 (`30pct`), 0.50 (`50pct`), 1.00 (`100pct`)
   - rolling: frac 1.00 (`results/work/100pct_rolling`)
   - `10pct_rolling` is explicitly **out of scope**; `results/rolling` final artifacts remain untouched.
4. **Approach**: A — new class in the benchmark suite + `--models` filter on the existing runner.
5. **Integration depth**: full — merge rows, regenerate Table 2, regenerate 100pct WRMSSE stats/CD diagrams, rerun consistency checks, refresh `results/holdout` (which is a copy of the 10pct work dir).
6. **Execution location**: local Windows machine. DATA_DIR = `E:\agentic\m5\m5proj_paper_experiment_store_sku_sampled\m5_data`; per-fraction caches under this repo at `cache/<frac>` (`cache/10pct`, `cache/30pct`, `cache/50pct`, `cache/100pct`, `cache/100pct_rolling`). Caches must be rebuilt locally via `warm_m5_cache.py`.

## Environment facts (verified)

- Benchmark suite lives in `scripts/m5_benchmarks.py` (`build_benchmark_suite()`), run by `scripts/run_m5_full_benchmark.py` → `03_benchmark_baselines/all_{mode}_rows.csv` + `all_{mode}_series_rows.csv` (plus per-task subdirs).
- Stats: `scripts/run_m5_stat_tests.py` merges benchmark + proposed + ablation rows (both block-level and series-level). Tables: `scripts/make_m5_paper_tables.py` merges benchmark + proposed rows → `06_paper_tables/` (Table 2 point, Table 3 quantile, Table 4 ablation).
- Per-frac stats were run with `--metric wspl --analysis-level series`. Only the 100pct dir additionally has WRMSSE stats/CD diagrams, produced by `scripts/generate_cd_diagrams_100pct.py` (hardcoded to 100pct paths, with `MODEL_NAME_MAP` for paper names).
- `results/holdout` = copy of `results/work/10pct` final artifacts; `results/rolling` = copy of `results/work/10pct_rolling`.
- Original split args (identical across all settings): `--t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 --gap 0 --min-train-origins 20 --min-final-train-origins 20 --sku-universe-mode history --store-sku-sample-seed 42`, frac varies.
- Consistency checker (`check_experiment_consistency.py`) requires `BottomUpGlobalHistGB` present and `BottomUpSeasonalNaive` absent in benchmark outputs; adding a new model does not violate it.
- Local `cache/m5_cache` is only a partial store_dept cache from an earlier sanity run — not usable for the 5 settings.

## Design

### 1. `scripts/m5_benchmarks.py` — new model

**Refactor** `BottomUpGlobalBoostingBenchmark`:
- Add `predict_child(samples) -> List[np.ndarray]`: fits nothing; builds child-level features via `build_bottom_level_global_features`, predicts per child, clips at 0 (existing `clip_predictions` behavior), returns one `[N_i, H]` block per sample.
- `predict_point(samples)` becomes: sum each block over children, stack → unchanged output for existing results.

**New class** `OLSReconciledHistGB(BenchmarkBase)`:
- `__init__(self, random_state=42)` composes:
  - `self.top_model = SklearnMultiOutputBenchmark(feature_builder=build_direct_aggregate_matrix, base_regressor=make_histgb_multioutput(max_depth=6, learning_rate=0.05, max_iter=300, random_state=random_state))` — identical config to `AggregateHistGB` in the suite.
  - `self.bottom_model = BottomUpGlobalBoostingBenchmark(random_state=random_state)` — identical config to `BottomUpGlobalHistGB` in the suite.
- `fit(train_samples)`: fits both base models on the same training samples.
- `predict_point(samples)`: for each sample, `ŷ_top = top_model.predict_point([s])`, child blocks from `bottom_model.predict_child([s])`; per horizon step h compute `ŷ̃_top[h] = (N·ŷ_top[h] + Σᵢ ŷ_childᵢ[h]) / (N+1)`; N == 0 guard returns ŷ_top. Stack to `[S, H]`.
- Quantiles: not implemented beyond `BenchmarkBase` defaults; the runner is invoked with `--quantiles ""` so no WSPL rows are ever produced for this model.

**Registration**:
- `build_benchmark_suite()` gains `"Reconcilation": OLSReconciledHistGB(random_state=random_state)`.
- `clone_benchmark()` gains a branch returning a fresh `OLSReconciledHistGB` with the same random_state (needed per rolling fold).

### 2. `scripts/run_m5_full_benchmark.py` — model filter

- New CLI arg `--models` (comma-separated names, default empty = run whole suite). After `build_benchmark_suite()`, keep only requested models; error if a requested name is not in the suite.
- No other changes. Running with `--models Reconcilation --quantiles ""` writes CSVs containing only the new model's point rows (`wrmsse`, `wape`) and series rows (`rmsse`).

### 3. Execution plan (local, Windows bash)

Set `PYTHONUNBUFFERED=1` and the single-thread env vars before every run (per CLAUDE.md).

**Step 3.1 — warm caches** (one-time; 100pct is the slow one):

```bash
DATA_DIR="E:/agentic/m5/m5proj_paper_experiment_store_sku_sampled/m5_data"
SPLIT="--t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 --gap 0 --min-train-origins 20 --min-final-train-origins 20 --sku-universe-mode history --store-sku-sample-seed 42"

# holdout caches
python scripts/warm_m5_cache.py --data-dir "$DATA_DIR" --cache-dir cache/10pct --tasks store_dept,store_cat,state_dept --holdout --store-sku-sample-frac 0.10 $SPLIT
python scripts/warm_m5_cache.py --data-dir "$DATA_DIR" --cache-dir cache/30pct --tasks store_dept,store_cat,state_dept --holdout --store-sku-sample-frac 0.30 $SPLIT
python scripts/warm_m5_cache.py --data-dir "$DATA_DIR" --cache-dir cache/50pct --tasks store_dept,store_cat,state_dept --holdout --store-sku-sample-frac 0.50 $SPLIT
python scripts/warm_m5_cache.py --data-dir "$DATA_DIR" --cache-dir cache/100pct --tasks store_dept,store_cat,state_dept --holdout --store-sku-sample-frac 1.00 $SPLIT
# rolling cache (frac 1.0, holdout+rolling)
python scripts/warm_m5_cache.py --data-dir "$DATA_DIR" --cache-dir cache/100pct_rolling --tasks store_dept,store_cat,state_dept --holdout --rolling --store-sku-sample-frac 1.00 $SPLIT
```

**Step 3.2 — run the new benchmark only** (output isolated in `recon/` subdirs):

```bash
# holdout
python scripts/run_m5_full_benchmark.py --data-dir "$DATA_DIR" --cache-dir cache/10pct \
  --output-dir results/work/10pct/03_benchmark_baselines/recon --mode holdout \
  --tasks store_dept,store_cat,state_dept --models Reconcilation --quantiles "" \
  --store-sku-sample-frac 0.10 $SPLIT

python scripts/run_m5_full_benchmark.py --data-dir "$DATA_DIR" --cache-dir cache/30pct \
  --output-dir results/work/30pct/03_benchmark_baselines/recon --mode holdout \
  --tasks store_dept,store_cat,state_dept --models Reconcilation --quantiles "" \
  --store-sku-sample-frac 0.30 $SPLIT

python scripts/run_m5_full_benchmark.py --data-dir "$DATA_DIR" --cache-dir cache/50pct \
  --output-dir results/work/50pct/03_benchmark_baselines/recon --mode holdout \
  --tasks store_dept,store_cat,state_dept --models Reconcilation --quantiles "" \
  --store-sku-sample-frac 0.50 $SPLIT

python scripts/run_m5_full_benchmark.py --data-dir "$DATA_DIR" --cache-dir cache/100pct \
  --output-dir results/work/100pct/03_benchmark_baselines/recon --mode holdout \
  --tasks store_dept,store_cat,state_dept --models Reconcilation --quantiles "" \
  --store-sku-sample-frac 1.00 $SPLIT

# rolling (frac 1.0)
python scripts/run_m5_full_benchmark.py --data-dir "$DATA_DIR" --cache-dir cache/100pct_rolling \
  --output-dir results/work/100pct_rolling/03_benchmark_baselines/recon --mode rolling \
  --tasks store_dept,store_cat,state_dept --models Reconcilation --quantiles "" \
  --store-sku-sample-frac 1.00 $SPLIT
```

**Step 3.3 — merge & regenerate** (new `scripts/merge_recon_results.py`, idempotent; run per work dir):
- Append `Reconcilation` rows from `recon/all_{mode}_rows.csv` into `03_benchmark_baselines/all_{mode}_rows.csv` (skip if model already present in the target).
- Append `recon/all_{mode}_series_rows.csv` into `all_{mode}_series_rows.csv` (same guard).
- Rerun `make_m5_paper_tables.py` with the existing benchmark/proposed/ablation row paths → Table 2 gains the new model; Table 3/4 content unchanged.
- Rerun `run_m5_stat_tests.py --metric wspl --analysis-level series --series-block-cols task,agg_id --ablation-models M0_AggHistOnly,M1_AggHistFutureSummary` (WSPL outputs unchanged since the new model has no quantile rows; combined CSVs refreshed).
- 100pct only: add `"Reconcilation": "Reconcilation"` identity entry to `MODEL_NAME_MAP` in `generate_cd_diagrams_100pct.py`, rerun it → WRMSSE CD diagram/stat tables now include the new model.
- Rerun `check_experiment_consistency.py --mode <mode> --metric wspl` per work dir.
- Refresh `results/holdout` from the 10pct work dir: copy `06_paper_tables/Table2_main_point.tex`, `Table2_main_point.csv`, `point_summary.csv`, `combined_rows.csv`, `03_benchmark_baselines/point_summary_holdout.csv` (exact file list finalized during implementation). `results/rolling` untouched.

### 4. Verification

1. **Formula unit check**: tiny synthetic sample — N children with known ŷ_child and ŷ_top; assert `(N·ŷ_top + Σŷ_child)/(N+1)` and the N==0 guard.
2. **Dry run**: 10pct holdout, `--tasks store_dept` only, `--models Reconcilation`, inspect output rows and one sample's reconciled values vs. hand-computed values.
3. **Consistency**: `check_experiment_consistency.py` passes for each of the 5 work dirs after merge.
4. **Diff-check existing results**: merging must not alter any pre-existing row (compare row counts and checksums of the original CSVs before/after — only appended rows allowed).

## Risks / costs

- **Local cache rebuild**: 100pct cache build is the slowest step (~30-60+ min); 100pct bottom-up HistGB training per task is the heaviest benchmark step; rolling 100pct repeats it per fold (5 folds × 3 tasks).
- **Name spelling**: `Reconcilation` (user's exact spelling) is used everywhere, including LaTeX tables and CD diagrams.
- **No quantile rows**: the new model never appears in WSPL stats or Table 3; the paper's WSPL statistical analysis is unchanged.

## Out of scope

- Quantile (WSPL) support for the new model.
- `10pct_rolling` work dir and `results/rolling` artifacts.
- Re-running any existing model or experiment.
