# Reconcilation Benchmark Implementation & Experiment Run Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a point-only OLS-reconciled benchmark (`Reconcilation`) that combines AggregateHistGB (top) and BottomUpGlobalHistGB (child) base forecasts, run it for the 5 experiment settings (holdout frac 0.1/0.3/0.5/1.0 + rolling frac 1.0), and merge its rows into the existing results with full stats/table regeneration.

**Architecture:** New `OLSReconciledHistGB` class in `scripts/m5_benchmarks.py` composes two existing benchmarks and reconciles per sample per horizon step via `(N·ŷ_top[h] + Σᵢŷ_childᵢ[h]) / (N+1)`. A `--models` filter on the existing runner lets us run only the new model. A new idempotent merge script appends rows and reruns tables/stats/consistency per work dir.

**Tech Stack:** Python 3.11, numpy, pandas, sklearn (HistGB/ElasticNet), threadpoolctl. Windows bash shell. No pytest — tests are plain-Python assert scripts run with `python`.

## Global Constraints

- Model name is exactly `Reconcilation` (user's spelling) in all CSVs, stats, tables, and CD diagrams.
- Reconciliation formula per sample per horizon step h: `ŷ̃_top[h] = (N·ŷ_top[h] + Σᵢ ŷ_childᵢ[h]) / (N+1)`; guard: if N == 0, `ŷ̃_top = ŷ_top`.
- The new model is point-only: every run uses `--quantiles ""`. WSPL stats and Table 3 must remain unchanged.
- Base-model configs must mirror the suite: top = HistGB multioutput (max_depth=6, learning_rate=0.05, max_iter=300, random_state=42); bottom = `BottomUpGlobalBoostingBenchmark` defaults (clip_predictions=True, random_state=42).
- Split args identical to original experiments (verbatim): `--t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 --gap 0 --min-train-origins 20 --min-final-train-origins 20 --sku-universe-mode history --store-sku-sample-seed 42`.
- DATA_DIR = `E:/agentic/m5/m5proj_paper_experiment_store_sku_sampled/m5_data`; caches per fraction under repo `cache/<frac>` (`cache/10pct`, `cache/30pct`, `cache/50pct`, `cache/100pct`, `cache/100pct_rolling`). Never mix fractions in one cache dir (cache files are keyed only by task+split name, not config).
- Env vars before every Python run: `export PYTHONUNBUFFERED=1` and `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`.
- Merging must never alter existing rows — only append `Reconcilation` rows; merges are idempotent (skip if model already present).
- `10pct_rolling` and `results/rolling` are out of scope — do not touch them.

---

### Task 1: Refactor `BottomUpGlobalBoostingBenchmark` with `predict_child()`

**Files:**
- Modify: `scripts/m5_benchmarks.py:465-482` (predict_point)
- Test: `tests/test_ols_reconciled.py` (new)

**Interfaces:**
- Produces: `BottomUpGlobalBoostingBenchmark.predict_child(samples) -> List[np.ndarray]` (one `[N_i, H]` float32 block per sample, clipped at 0 if `clip_predictions`); `predict_point(samples) -> np.ndarray` unchanged behavior (sum of blocks).

- [ ] **Step 1: Write the failing test**

Create `tests/test_ols_reconciled.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import numpy as np

import m5_benchmarks as mb
from m5_benchmarks import BottomUpGlobalBoostingBenchmark


class FakeRegressor:
    def predict(self, X):
        return np.arange(X.shape[0] * 2, dtype=np.float32).reshape(X.shape[0], 2)


def test_predict_child_returns_blocks_and_point_sums_them():
    mb.build_bottom_level_global_features = lambda samples: (np.zeros((5, 3), dtype=np.float32), [2, 3])
    bench = BottomUpGlobalBoostingBenchmark(random_state=42, clip_predictions=False)
    bench.model_ = FakeRegressor()
    blocks = bench.predict_child([{}, {}])
    assert len(blocks) == 2, blocks
    assert blocks[0].shape == (2, 2) and blocks[1].shape == (3, 2), [b.shape for b in blocks]
    point = bench.predict_point([{}, {}])
    expected = np.vstack([blocks[0].sum(axis=0), blocks[1].sum(axis=0)])
    np.testing.assert_allclose(point, expected)
    print("test_predict_child_returns_blocks_and_point_sums_them PASS")


if __name__ == "__main__":
    test_predict_child_returns_blocks_and_point_sums_them()
    print("ALL TESTS PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_ols_reconciled.py`
Expected: FAIL with `AttributeError: 'BottomUpGlobalBoostingBenchmark' object has no attribute 'predict_child'`

- [ ] **Step 3: Implement `predict_child`**

In `scripts/m5_benchmarks.py`, replace the existing `predict_point` method of `BottomUpGlobalBoostingBenchmark` (lines 465-482) with:

```python
    def predict_child(self, samples: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        X, child_counts = build_bottom_level_global_features(samples)
        with threadpool_limits(limits=1):
            child_pred = np.asarray(self.model_.predict(X), dtype=np.float32)
        if child_pred.ndim == 1:
            child_pred = child_pred.reshape(-1, 1)
        if self.clip_predictions:
            child_pred = np.maximum(child_pred, 0.0)

        blocks: List[np.ndarray] = []
        offset = 0
        for n_child in child_counts:
            blocks.append(child_pred[offset: offset + n_child])
            offset += n_child
        return blocks

    def predict_point(self, samples: Sequence[Dict[str, Any]]) -> np.ndarray:
        blocks = self.predict_child(samples)
        return np.vstack([b.sum(axis=0) for b in blocks]).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_ols_reconciled.py`
Expected: `test_predict_child_returns_blocks_and_point_sums_them PASS` then `ALL TESTS PASS`

- [ ] **Step 5: Commit**

```bash
git add scripts/m5_benchmarks.py tests/test_ols_reconciled.py
git commit -m "refactor: expose predict_child on BottomUpGlobalBoostingBenchmark"
```

---

### Task 2: Add `OLSReconciledHistGB` and register as `Reconcilation`

**Files:**
- Modify: `scripts/m5_benchmarks.py` (new class after `BottomUpGlobalBoostingBenchmark`, suite entry ~line 534, clone branch ~line 795)
- Test: `tests/test_ols_reconciled.py` (extend)

**Interfaces:**
- Produces: `OLSReconciledHistGB(random_state=42)` with `fit(train_samples)` and `predict_point(samples) -> np.ndarray` ([S, H] reconciled aggregate forecasts); suite key `"Reconcilation"`; `clone_benchmark` support.

- [ ] **Step 1: Extend the failing test**

Append to `tests/test_ols_reconciled.py` (and update the `__main__` block):

```python
from m5_benchmarks import OLSReconciledHistGB, build_benchmark_suite, clone_benchmark


class FakeTop:
    def fit(self, samples):
        return self

    def predict_point(self, samples):
        return np.array([[10.0, 20.0]], dtype=np.float32)


class FakeBottom:
    def fit(self, samples):
        return self

    def predict_child(self, samples):
        return [np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)]


class FakeBottomEmpty:
    def fit(self, samples):
        return self

    def predict_child(self, samples):
        return [np.zeros((0, 2), dtype=np.float32)]


def test_ols_reconciliation_formula():
    bench = OLSReconciledHistGB(random_state=42)
    bench.top_model = FakeTop()
    bench.bottom_model = FakeBottom()
    preds = bench.predict_point([{}])
    # N=3: (3*[10,20] + [6,6]) / 4 = [9.0, 16.5]
    np.testing.assert_allclose(preds, [[9.0, 16.5]])
    print("test_ols_reconciliation_formula PASS")


def test_ols_reconciliation_zero_children_guard():
    bench = OLSReconciledHistGB(random_state=42)
    bench.top_model = FakeTop()
    bench.bottom_model = FakeBottomEmpty()
    preds = bench.predict_point([{}])
    np.testing.assert_allclose(preds, [[10.0, 20.0]])
    print("test_ols_reconciliation_zero_children_guard PASS")


def test_suite_and_clone():
    suite = build_benchmark_suite(random_state=42)
    assert "Reconcilation" in suite, list(suite)
    cloned = clone_benchmark(suite["Reconcilation"])
    assert isinstance(cloned, OLSReconciledHistGB)
    assert cloned.random_state == 42
    print("test_suite_and_clone PASS")
```

Update the `__main__` block to:

```python
if __name__ == "__main__":
    test_predict_child_returns_blocks_and_point_sums_them()
    test_ols_reconciliation_formula()
    test_ols_reconciliation_zero_children_guard()
    test_suite_and_clone()
    print("ALL TESTS PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_ols_reconciled.py`
Expected: FAIL with `ImportError: cannot import name 'OLSReconciledHistGB'`

- [ ] **Step 3: Implement the class**

In `scripts/m5_benchmarks.py`, insert after the `BottomUpGlobalBoostingBenchmark` class (before the `# 5) Factory functions` section):

```python
class OLSReconciledHistGB(BenchmarkBase):
    """
    Standard OLS hierarchical reconciliation of aggregate-level (AggregateHistGB)
    and bottom-level (BottomUpGlobalHistGB) base forecasts.

    Per sample and horizon step h, the reconciled aggregate forecast is the
    closed form of the OLS projection S(S'S)^-1 S' y on the two-level hierarchy
    [top; children], summed back to the aggregate level:

        y_tilde[h] = (N * y_top[h] + sum_i y_child_i[h]) / (N + 1)

    with N the number of child store-SKUs in the sample. If N == 0 the
    aggregate base forecast is returned unchanged. Point forecasts only.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = int(random_state)
        self.top_model = SklearnMultiOutputBenchmark(
            feature_builder=build_direct_aggregate_matrix,
            base_regressor=make_histgb_multioutput(
                max_depth=6, learning_rate=0.05, max_iter=300, random_state=random_state
            ),
        )
        self.bottom_model = BottomUpGlobalBoostingBenchmark(random_state=random_state)

    def fit(self, train_samples: Sequence[Dict[str, Any]]) -> "OLSReconciledHistGB":
        self.top_model.fit(train_samples)
        self.bottom_model.fit(train_samples)
        return self

    def predict_point(self, samples: Sequence[Dict[str, Any]]) -> np.ndarray:
        preds: List[np.ndarray] = []
        for s in samples:
            y_top = np.asarray(self.top_model.predict_point([s])[0], dtype=np.float32)  # [H]
            child_blocks = self.bottom_model.predict_child([s])
            if not child_blocks or child_blocks[0].shape[0] == 0:
                preds.append(y_top)
                continue
            y_child = child_blocks[0]  # [N, H]
            n_child = int(y_child.shape[0])
            y_bu = y_child.sum(axis=0)  # [H]
            preds.append(((n_child * y_top) + y_bu) / float(n_child + 1))
        return np.vstack(preds).astype(np.float32)
```

- [ ] **Step 4: Register in the suite**

In `build_benchmark_suite()` (line ~534), after the `"BottomUpGlobalHistGB"` entry, add:

```python
        # OLS-reconciled combination of the aggregate-level and bottom-up HistGB forecasts
        "Reconcilation": OLSReconciledHistGB(random_state=random_state),
```

- [ ] **Step 5: Add the clone branch**

In `clone_benchmark()` (after the `BottomUpGlobalBoostingBenchmark` branch, before `raise TypeError`), add:

```python
    if isinstance(benchmark, OLSReconciledHistGB):
        return OLSReconciledHistGB(random_state=benchmark.random_state)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python tests/test_ols_reconciled.py`
Expected: four `... PASS` lines then `ALL TESTS PASS`

- [ ] **Step 7: Commit**

```bash
git add scripts/m5_benchmarks.py tests/test_ols_reconciled.py
git commit -m "feat: add OLSReconciledHistGB benchmark (Reconcilation)"
```

---

### Task 3: Add `--models` filter to `run_m5_full_benchmark.py`

**Files:**
- Modify: `scripts/run_m5_full_benchmark.py:97-98` (suite build) and `:165-195` (argparser)

**Interfaces:**
- Produces: CLI flag `--models "a,b"` (default `""` = full suite); unknown names raise `ValueError`.

- [ ] **Step 1: Add the argparse flag**

After the `--tasks` argument (line ~169) add:

```python
    parser.add_argument("--models", type=str, default="", help="Comma-separated benchmark model names to run. Empty string runs the full suite.")
```

- [ ] **Step 2: Apply the filter after suite construction**

Replace the two lines:

```python
    print("Building benchmark suite...")
    suite = build_benchmark_suite(random_state=args.random_state)
```

with:

```python
    print("Building benchmark suite...")
    suite = build_benchmark_suite(random_state=args.random_state)

    model_filter = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if model_filter:
        unknown = [m for m in model_filter if m not in suite]
        if unknown:
            raise ValueError(f"Unknown benchmark model(s): {unknown}. Available: {sorted(suite)}")
        suite = {m: suite[m] for m in model_filter}
        print(f"Running selected benchmark models: {model_filter}")
```

- [ ] **Step 3: Smoke-test the filter**

Run:

```bash
export PYTHONUNBUFFERED=1
python scripts/run_m5_full_benchmark.py --help
python -c "
import sys; sys.path.insert(0, 'scripts')
import argparse
import run_m5_full_benchmark as m
print('import ok')
"
```

Expected: `--help` lists `--models`; import has no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_m5_full_benchmark.py
git commit -m "feat: add --models filter to full benchmark runner"
```

---

### Task 4: Add `Reconcilation` to `MODEL_NAME_MAP` in `generate_cd_diagrams_100pct.py`

**Files:**
- Modify: `scripts/generate_cd_diagrams_100pct.py:41` (after `"SeasonalNaive"` entry)

- [ ] **Step 1: Add the mapping entry**

After the line `"SeasonalNaive": "Seasonal Naive",` add:

```python
    "Reconcilation": "Reconcilation",
```

- [ ] **Step 2: Verify import**

Run: `python -c "import sys; sys.path.insert(0, 'scripts'); import generate_cd_diagrams_100pct as g; assert 'Reconcilation' in g.MODEL_NAME_MAP; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_cd_diagrams_100pct.py
git commit -m "feat: add Reconcilation to CD diagram model name map"
```

---

### Task 5: Create `scripts/merge_recon_results.py`

**Files:**
- Create: `scripts/merge_recon_results.py`

**Interfaces:**
- Consumes: `results/work/<dir>/03_benchmark_baselines/recon/all_{mode}_rows.csv` and `all_{mode}_series_rows.csv` (written by the runner).
- Produces: appends `Reconcilation` rows into the parent `all_{mode}_rows.csv` / `all_{mode}_series_rows.csv` (idempotent), reruns `make_m5_paper_tables.py`, `run_m5_stat_tests.py --metric wspl`, and `check_experiment_consistency.py` via subprocess.

- [ ] **Step 1: Write the script**

```python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

MODEL = "Reconcilation"


def append_new_model(recon_csv: Path, target_csv: Path) -> str:
    """Append Reconcilation rows from recon_csv into target_csv. Idempotent."""
    if not recon_csv.exists():
        raise FileNotFoundError(f"Missing recon CSV: {recon_csv}")
    recon = pd.read_csv(recon_csv)
    if recon.empty:
        return "empty recon csv"
    recon = recon.loc[recon["model"].astype(str) == MODEL].copy()
    if recon.empty:
        raise ValueError(f"No {MODEL} rows found in {recon_csv}")
    if target_csv.exists():
        target = pd.read_csv(target_csv)
        n_before = int(len(target))
        if MODEL in set(target["model"].astype(str)):
            return f"already present ({n_before} rows untouched)"
    else:
        target = pd.DataFrame()
        n_before = 0
    merged = pd.concat([target, recon], axis=0, ignore_index=True)
    merged.to_csv(target_csv, index=False)
    return f"appended {len(recon)} rows (was {n_before})"


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Reconcilation benchmark rows into existing experiment CSVs and regenerate stats/tables."
    )
    parser.add_argument("--work-dir", required=True, help="e.g. results/work/10pct")
    parser.add_argument("--mode", required=True, choices=["holdout", "rolling"])
    args = parser.parse_args()

    root = Path(args.work_dir)
    scripts = Path(__file__).resolve().parent
    bench = root / "03_benchmark_baselines"
    recon_dir = bench / "recon"
    mode = args.mode

    print(f"== Merging {MODEL} rows for {root} ({mode}) ==")
    print(f"  rows: {append_new_model(recon_dir / f'all_{mode}_rows.csv', bench / f'all_{mode}_rows.csv')}")
    print(f"  series rows: {append_new_model(recon_dir / f'all_{mode}_series_rows.csv', bench / f'all_{mode}_series_rows.csv')}")

    benchmark_rows = str(bench / f"all_{mode}_rows.csv")
    benchmark_series = str(bench / f"all_{mode}_series_rows.csv")
    proposed_rows = str(root / "04_benchmark_proposed" / f"proposed_all_{mode}_rows.csv")
    proposed_series = str(root / "04_benchmark_proposed" / f"proposed_all_{mode}_series_rows.csv")
    ablation_rows = str(root / "02_ablation" / f"all_ablation_rows_{mode}.csv")
    ablation_series = str(root / "02_ablation" / f"all_ablation_series_rows_{mode}.csv")
    tables_dir = root / "06_paper_tables"
    stats_dir = root / "05_stat_tests"
    tables_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print("== Regenerating paper tables ==")
    run_cmd([
        sys.executable, str(scripts / "make_m5_paper_tables.py"),
        "--benchmark-rows", benchmark_rows,
        "--proposed-rows", proposed_rows,
        "--ablation-rows", ablation_rows,
        "--output-dir", str(tables_dir),
    ])

    print("== Regenerating WSPL stat tests ==")
    run_cmd([
        sys.executable, str(scripts / "run_m5_stat_tests.py"),
        "--benchmark-rows", benchmark_rows,
        "--proposed-rows", proposed_rows,
        "--ablation-rows", ablation_rows,
        "--benchmark-series-rows", benchmark_series,
        "--proposed-series-rows", proposed_series,
        "--ablation-series-rows", ablation_series,
        "--ablation-models", "M0_AggHistOnly,M1_AggHistFutureSummary",
        "--metric", "wspl",
        "--analysis-level", "series",
        "--series-block-cols", "task,agg_id",
        "--output-dir", str(stats_dir),
    ])

    print("== Consistency check ==")
    run_cmd([
        sys.executable, str(scripts / "check_experiment_consistency.py"),
        "--work-dir", str(root), "--mode", mode, "--metric", "wspl",
    ])

    print(f"Done. Merged {MODEL} into {root} and regenerated tables/stats.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test help**

Run: `python scripts/merge_recon_results.py --help`
Expected: usage text with `--work-dir` and `--mode`

- [ ] **Step 3: Commit**

```bash
git add scripts/merge_recon_results.py
git commit -m "feat: add merge_recon_results.py for Reconcilation result integration"
```

---

### Task 6: Warm the 10pct cache and dry-run store_dept only

**Files:** none (execution only)

- [ ] **Step 1: Warm cache/10pct**

```bash
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/warm_m5_cache.py --data-dir "E:/agentic/m5/m5proj_paper_experiment_store_sku_sampled/m5_data" \
  --cache-dir cache/10pct --tasks store_dept,store_cat,state_dept --holdout \
  --store-sku-sample-frac 0.10 \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 --gap 0 \
  --min-train-origins 20 --min-final-train-origins 20 --sku-universe-mode history \
  --store-sku-sample-seed 42 2>&1 | tee results/work/10pct/warm_10pct_recon.log
```

Expected: completes and `cache/10pct/task__store_dept/final_train_samples.pkl`, `test_samples.pkl` exist.

- [ ] **Step 2: Dry-run the new benchmark on store_dept only**

```bash
python scripts/run_m5_full_benchmark.py --data-dir "E:/agentic/m5/m5proj_paper_experiment_store_sku_sampled/m5_data" \
  --cache-dir cache/10pct \
  --output-dir results/work/10pct/03_benchmark_baselines/recon --mode holdout \
  --tasks store_dept --models Reconcilation --quantiles "" \
  --store-sku-sample-frac 0.10 \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 --gap 0 \
  --min-train-origins 20 --min-final-train-origins 20 --sku-universe-mode history \
  --store-sku-sample-seed 42 2>&1 | tee results/work/10pct/recon_dryrun_store_dept.log
```

Expected: completes; `results/work/10pct/03_benchmark_baselines/recon/all_holdout_rows.csv` contains only `Reconcilation` rows with non-null `wrmsse` and null `wspl`; series CSV has `metric == rmsse` rows for `Reconcilation`.

- [ ] **Step 3: Spot-check reconciliation on one sample**

```bash
python - <<'EOF'
import pandas as pd, numpy as np
rows = pd.read_csv("results/work/10pct/03_benchmark_baselines/recon/all_holdout_rows.csv")
print(rows[["task","model","fold","wrmsse","wape","wspl"]])
ser = pd.read_csv("results/work/10pct/03_benchmark_baselines/recon/all_holdout_series_rows.csv")
print(ser["model"].value_counts())
EOF
```

Expected: one `Reconcilation` row per task with `wrmsse` in a plausible range (0.7–1.1 for 10pct store_dept) and `wspl` NaN; series rows only for `Reconcilation` with `metric == rmsse`.

- [ ] **Step 4: Commit the dry-run outputs are intentionally NOT committed (recon/ dirs stay untracked); commit nothing here.**

---

### Task 7: Warm remaining caches (30pct, 50pct, 100pct, 100pct_rolling)

**Files:** none (execution only, long-running — run in background sequentially)

- [ ] **Step 1: Warm cache/30pct**

```bash
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/warm_m5_cache.py --data-dir "E:/agentic/m5/m5proj_paper_experiment_store_sku_sampled/m5_data" \
  --cache-dir cache/30pct --tasks store_dept,store_cat,state_dept --holdout \
  --store-sku-sample-frac 0.30 \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 --gap 0 \
  --min-train-origins 20 --min-final-train-origins 20 --sku-universe-mode history \
  --store-sku-sample-seed 42 2>&1 | tee results/work/30pct/warm_30pct_recon.log
```

Expected: completes; `cache/30pct/task__store_dept/task_package.pkl` etc. exist.

- [ ] **Step 2: Warm cache/50pct**

Same command with `--cache-dir cache/50pct --store-sku-sample-frac 0.50` and log `results/work/50pct/warm_50pct_recon.log`.

- [ ] **Step 3: Warm cache/100pct (holdout)**

Same command with `--cache-dir cache/100pct --store-sku-sample-frac 1.00` and log `results/work/100pct/warm_100pct_recon.log`. This is the slowest warm (expect 30–90 min).

- [ ] **Step 4: Warm cache/100pct_rolling (holdout + rolling)**

Same command with `--cache-dir cache/100pct_rolling --store-sku-sample-frac 1.00 --holdout --rolling` and log `results/work/100pct_rolling/warm_100pct_rolling_recon.log`. Expect another 30–90 min.

- [ ] **Step 5: Verify all caches**

```bash
for d in 10pct 30pct 50pct 100pct; do ls cache/$d/task__store_dept/test_samples.pkl >/dev/null && echo "$d holdout ok"; done
ls cache/100pct_rolling/task__store_dept/rolling_fold_1_valid_samples.pkl >/dev/null && echo "100pct_rolling rolling ok"
```

Expected: all five lines print `ok`.

---

### Task 8: Run the new benchmark for the 5 experiment settings

**Files:** none (execution only, long-running — run in background sequentially)

For each run: output goes to `results/work/<dir>/03_benchmark_baselines/recon/`; `--models Reconcilation --quantiles ""`.

- [ ] **Step 1: holdout 10pct (all 3 tasks; overwrites the dry-run recon dir)**

```bash
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/run_m5_full_benchmark.py --data-dir "E:/agentic/m5/m5proj_paper_experiment_store_sku_sampled/m5_data" \
  --cache-dir cache/10pct \
  --output-dir results/work/10pct/03_benchmark_baselines/recon --mode holdout \
  --tasks store_dept,store_cat,state_dept --models Reconcilation --quantiles "" \
  --store-sku-sample-frac 0.10 \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 --gap 0 \
  --min-train-origins 20 --min-final-train-origins 20 --sku-universe-mode history \
  --store-sku-sample-seed 42 2>&1 | tee results/work/10pct/recon_holdout_10pct.log
```

Expected: 3 task rows for `Reconcilation` in `all_holdout_rows.csv` (plus one row per task in per-task subdirs).

- [ ] **Step 2: holdout 30pct**

Same command with `--cache-dir cache/30pct --output-dir results/work/30pct/03_benchmark_baselines/recon --store-sku-sample-frac 0.30`, log `results/work/30pct/recon_holdout_30pct.log`.

- [ ] **Step 3: holdout 50pct**

Same command with `--cache-dir cache/50pct --output-dir results/work/50pct/03_benchmark_baselines/recon --store-sku-sample-frac 0.50`, log `results/work/50pct/recon_holdout_50pct.log`.

- [ ] **Step 4: holdout 100pct**

Same command with `--cache-dir cache/100pct --output-dir results/work/100pct/03_benchmark_baselines/recon --store-sku-sample-frac 1.00`, log `results/work/100pct/recon_holdout_100pct.log`. Slowest run (bottom-up HistGB on the full 100% SKU sample).

- [ ] **Step 5: rolling 100pct**

Same command with `--cache-dir cache/100pct_rolling --output-dir results/work/100pct_rolling/03_benchmark_baselines/recon --mode rolling --store-sku-sample-frac 1.00`, log `results/work/100pct_rolling/recon_rolling_100pct.log`. Expect 5 folds per task; folds must be 1..5.

- [ ] **Step 6: Verify all 5 recon outputs**

```bash
for d in 10pct 30pct 50pct 100pct; do
  echo "== $d =="; python -c "import pandas as pd; df=pd.read_csv('results/work/$d/03_benchmark_baselines/recon/all_holdout_rows.csv'); print(df.groupby('task')['wrmsse'].count())"
done
echo "== 100pct_rolling =="; python -c "import pandas as pd; df=pd.read_csv('results/work/100pct_rolling/03_benchmark_baselines/recon/all_rolling_rows.csv'); print(df.groupby(['task','fold'])['wrmsse'].count())"
```

Expected: 1 `Reconcilation` wrmsse row per task for each holdout dir; 1 row per (task, fold) with folds 1–5 for rolling.

---

### Task 9: Merge and regenerate per work dir

**Files:** none new (runs `scripts/merge_recon_results.py`, `scripts/generate_cd_diagrams_100pct.py`, copies)

- [ ] **Step 1: Merge + regen for 10pct**

```bash
export PYTHONUNBUFFERED=1
python scripts/merge_recon_results.py --work-dir results/work/10pct --mode holdout
```

Expected: prints "appended ... rows" for rows and series rows; tables/stats/consistency subprocesses complete; `Consistency check passed.` in output.

- [ ] **Step 2: Merge + regen for 30pct, 50pct, 100pct (holdout)**

Run the same command with `--work-dir results/work/30pct`, `results/work/50pct`, `results/work/100pct` (mode holdout).

- [ ] **Step 3: Merge + regen for 100pct_rolling**

```bash
python scripts/merge_recon_results.py --work-dir results/work/100pct_rolling --mode rolling
```

- [ ] **Step 4: Idempotency check**

Rerun: `python scripts/merge_recon_results.py --work-dir results/work/10pct --mode holdout`
Expected: prints `already present (... rows untouched)` for both CSVs; tables/stats regenerate without error.

- [ ] **Step 5: Regenerate 100pct WRMSSE/CD diagrams**

```bash
python scripts/generate_cd_diagrams_100pct.py
```

Expected: rewrites `results/work/100pct/05_stat_tests/Figure_cd_diagram_wrmsse_series*.png` and `results/work/100pct_rolling/05_stat_tests/...` with `Reconcilation` present in `Table_stats_wrmsse_series*.csv`.

- [ ] **Step 6: Refresh `results/holdout` from the 10pct work dir**

```bash
cp results/work/10pct/06_paper_tables/Table2_main_point.tex results/holdout/Table2_main_point.tex
cp results/work/10pct/06_paper_tables/point_summary.csv results/holdout/point_summary.csv
cp results/work/10pct/03_benchmark_baselines/point_summary_holdout.csv results/holdout/point_summary_holdout.csv
```

Expected: `results/holdout/Table2_main_point.tex` now contains a `Reconcilation` row.

- [ ] **Step 7: Commit all regenerated artifacts**

```bash
git add -u results/
git status  # review that only intended tracked files changed (new logs and recon/ dirs stay untracked)
git commit -m "results: add Reconcilation benchmark rows and regenerate tables/stats"
```

---

### Task 10: Final verification

- [ ] **Step 1: Verify existing rows unchanged**

```bash
python - <<'EOF'
import pandas as pd
for d, mode in [("10pct","holdout"),("30pct","holdout"),("50pct","holdout"),("100pct","holdout"),("100pct_rolling","rolling")]:
    df = pd.read_csv(f"results/work/{d}/03_benchmark_baselines/all_{mode}_rows.csv")
    recon = df[df.model == "Reconcilation"]
    others = df[df.model != "Reconcilation"]
    print(d, "recon rows:", len(recon), "other rows:", len(others))
    assert recon["wspl"].isna().all(), "Reconcilation must have no wspl rows"
    assert others["model"].isin(["SeasonalNaive","AggregateElasticNet","AggregateHistGB","ChildSummaryElasticNet","ChildSummaryHistGB","BottomUpGlobalHistGB"]).all()
print("existing rows sanity OK")
EOF
```

Expected: prints 5 lines + `existing rows sanity OK`. Cross-check that `recon["wspl"].isna().all()` holds and the other model counts match the pre-merge counts recorded earlier (6 models × tasks × folds).

- [ ] **Step 2: Consistency checks all pass**

```bash
python scripts/check_experiment_consistency.py --work-dir results/work/10pct --mode holdout --metric wspl
python scripts/check_experiment_consistency.py --work-dir results/work/30pct --mode holdout --metric wspl
python scripts/check_experiment_consistency.py --work-dir results/work/50pct --mode holdout --metric wspl
python scripts/check_experiment_consistency.py --work-dir results/work/100pct --mode holdout --metric wspl
python scripts/check_experiment_consistency.py --work-dir results/work/100pct_rolling --mode rolling --metric wspl
```

Expected: each prints `Consistency check passed.`

- [ ] **Step 3: Inspect the new model's numbers**

```bash
python - <<'EOF'
import pandas as pd
for d, mode in [("10pct","holdout"),("30pct","holdout"),("50pct","holdout"),("100pct","holdout"),("100pct_rolling","rolling")]:
    df = pd.read_csv(f"results/work/{d}/06_paper_tables/point_summary.csv")
    print(f"== {d} ({mode}) ==")
    print(df[df.model == "Reconcilation"][["task","wrmsse_mean","wape_mean"]].to_string(index=False))
EOF
```

Expected: `Reconcilation` WRMSSE values in the same ballpark as the other HistGB models for each task (e.g., 0.75–0.95 on 10pct).

- [ ] **Step 4: Unit tests still pass**

Run: `python tests/test_ols_reconciled.py`
Expected: `ALL TESTS PASS`

- [ ] **Step 5: Final summary to user**

Report per-fraction reconciled WRMSSE vs `AggregateHistGB` and `BottomUpGlobalHistGB` (the two base models it combines) from the regenerated `point_summary.csv` files, plus the updated `results/holdout/Table2_main_point.tex`.
