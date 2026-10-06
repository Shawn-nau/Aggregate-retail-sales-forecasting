from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


REQUIRED_ROW_COLS = {"task", "model", "fold"}
REQUIRED_SERIES_COLS = {"task", "model", "fold", "agg_id", "metric", "metric_value"}
REQUIRED_ABLATION_MODELS = {"M0_AggHistOnly", "M1_AggHistFutureSummary"}
REQUIRED_BOTTOM_UP = "BottomUpGlobalHistGB"
REMOVED_BOTTOM_UP = "BottomUpSeasonalNaive"


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required CSV: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def assert_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise AssertionError(f"{name} is missing required columns: {missing}")


def assert_mode_folds(df: pd.DataFrame, mode: str, name: str) -> None:
    if df.empty or "fold" not in df.columns:
        return
    folds = pd.to_numeric(df["fold"], errors="coerce")
    if mode == "holdout":
        bad = sorted(set(folds.dropna().astype(int).tolist()) - {0})
        if bad:
            raise AssertionError(f"{name} contains non-zero folds under holdout mode: {bad[:10]}")
    elif mode == "rolling":
        if folds.dropna().empty:
            raise AssertionError(f"{name} has no fold values under rolling mode.")
        if (folds.dropna() <= 0).any():
            raise AssertionError(f"{name} contains non-positive fold ids under rolling mode.")
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def model_set(df: pd.DataFrame) -> set[str]:
    if df.empty or "model" not in df.columns:
        return set()
    return set(df["model"].dropna().astype(str).unique().tolist())


def assert_model_presence(rows: pd.DataFrame, series: pd.DataFrame, label: str, required: Sequence[str]) -> None:
    present = model_set(rows) | model_set(series)
    missing = sorted(set(required) - present)
    if missing:
        raise AssertionError(f"{label} is missing required model(s): {missing}")


def assert_model_absence(rows: pd.DataFrame, series: pd.DataFrame, label: str, removed: Sequence[str]) -> None:
    present = model_set(rows) | model_set(series)
    found = sorted(set(removed) & present)
    if found:
        raise AssertionError(f"{label} still contains removed model(s): {found}")


def assert_series_metric_coverage(series: pd.DataFrame, metric: str, name: str) -> None:
    if series.empty:
        raise AssertionError(f"{name} is empty; series-level statistical tests need series rows.")
    metric_map = {"wrmsse": "rmsse", "rmsse": "rmsse", "wspl": "mean_spl", "mean_spl": "mean_spl"}
    wanted = metric_map.get(metric, metric)
    if wanted not in set(series["metric"].astype(str)):
        raise AssertionError(f"{name} lacks metric='{wanted}' rows required for --metric {metric}.")


def check_complete_blocks(series_parts: List[pd.DataFrame], metric: str, ablation_models: Sequence[str]) -> tuple[int, int]:
    metric_map = {"wrmsse": "rmsse", "rmsse": "rmsse", "wspl": "mean_spl", "mean_spl": "mean_spl"}
    wanted = metric_map.get(metric, metric)
    filtered = []
    for i, df in enumerate(series_parts):
        if df.empty:
            continue
        sub = df.loc[df["metric"].astype(str) == wanted].copy()
        if i == 2 and ablation_models:
            sub = sub.loc[sub["model"].astype(str).isin(set(ablation_models))].copy()
        filtered.append(sub)
    if not filtered:
        raise AssertionError(f"No series rows found for metric={wanted}.")
    combo = pd.concat(filtered, axis=0, ignore_index=True)
    combo["block_id"] = combo[["task", "agg_id"]].astype(str).agg("__".join, axis=1)
    agg = combo.groupby(["block_id", "model"], as_index=False).agg(metric_value=("metric_value", "mean"))
    pivot = agg.pivot(index="block_id", columns="model", values="metric_value")
    complete = pivot.dropna(axis=0, how="any")
    if complete.empty:
        raise AssertionError("No complete task×agg_id blocks remain after aligning models for series-level stats.")
    if complete.shape[1] < 3:
        raise AssertionError(f"Only {complete.shape[1]} complete models found; Friedman/Nemenyi needs at least 3.")
    return int(complete.shape[0]), int(complete.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Check holdout/rolling output consistency for the M5 experiment workflow.")
    parser.add_argument("--work-dir", required=True, help="Workflow directory created by run_paper_experiment_plan.py")
    parser.add_argument("--mode", required=True, choices=["holdout", "rolling"])
    parser.add_argument("--metric", default="wspl", choices=["wrmsse", "rmsse", "wspl", "mean_spl"])
    parser.add_argument("--ablation-models", default="M0_AggHistOnly,M1_AggHistFutureSummary")
    args = parser.parse_args()

    root = Path(args.work_dir)
    paths = {
        "benchmark_rows": root / "03_benchmark_baselines" / f"all_{args.mode}_rows.csv",
        "benchmark_series": root / "03_benchmark_baselines" / f"all_{args.mode}_series_rows.csv",
        "proposed_rows": root / "04_benchmark_proposed" / f"proposed_all_{args.mode}_rows.csv",
        "proposed_series": root / "04_benchmark_proposed" / f"proposed_all_{args.mode}_series_rows.csv",
        "ablation_rows": root / "02_ablation" / f"all_ablation_rows_{args.mode}.csv",
        "ablation_series": root / "02_ablation" / f"all_ablation_series_rows_{args.mode}.csv",
    }

    benchmark_rows = read_csv(paths["benchmark_rows"])
    benchmark_series = read_csv(paths["benchmark_series"])
    proposed_rows = read_csv(paths["proposed_rows"])
    proposed_series = read_csv(paths["proposed_series"])
    ablation_rows = read_csv(paths["ablation_rows"])
    ablation_series = read_csv(paths["ablation_series"])

    for name, df in [("benchmark_rows", benchmark_rows), ("proposed_rows", proposed_rows), ("ablation_rows", ablation_rows)]:
        assert_columns(df, REQUIRED_ROW_COLS, name)
        assert_mode_folds(df, args.mode, name)
    for name, df in [("benchmark_series", benchmark_series), ("proposed_series", proposed_series), ("ablation_series", ablation_series)]:
        assert_columns(df, REQUIRED_SERIES_COLS, name)
        assert_mode_folds(df, args.mode, name)
        assert_series_metric_coverage(df, args.metric, name)

    assert_model_presence(benchmark_rows, benchmark_series, "benchmark outputs", [REQUIRED_BOTTOM_UP])
    assert_model_absence(benchmark_rows, benchmark_series, "benchmark outputs", [REMOVED_BOTTOM_UP])
    assert_model_presence(ablation_rows, ablation_series, "ablation outputs", sorted(REQUIRED_ABLATION_MODELS))

    ablation_models = [x.strip() for x in args.ablation_models.split(",") if x.strip()]
    n_blocks, n_models = check_complete_blocks(
        [benchmark_series, proposed_series, ablation_series],
        metric=args.metric,
        ablation_models=ablation_models,
    )

    print("Consistency check passed.")
    print(f"Mode: {args.mode}")
    print(f"Metric: {args.metric}")
    print(f"Complete task×agg_id blocks for series-level stats: {n_blocks}")
    print(f"Models in aligned matrix: {n_models}")
    print(f"Bottom-up benchmark present: {REQUIRED_BOTTOM_UP}")
    print(f"Removed bottom-up naive absent: {REMOVED_BOTTOM_UP}")
    print(f"Ablation models included: {', '.join(sorted(REQUIRED_ABLATION_MODELS))}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Consistency check failed: {exc}", file=sys.stderr)
        raise
