from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def maybe(flag: str, enabled: bool) -> List[str]:
    return [flag] if enabled else []


def split_control_args(args: argparse.Namespace) -> List[str]:
    """Arguments that must be identical across cache, tuning, ablation, benchmark, and proposed-model phases."""
    return [
        "--t-hist", str(args.t_hist),
        "--horizon", str(args.horizon),
        "--valid-size", str(args.valid_size),
        "--test-size", str(args.test_size),
        "--internal-valid-size", str(args.internal_valid_size),
        "--gap", str(args.gap),
        "--min-train-origins", str(args.min_train_origins),
        "--min-final-train-origins", str(args.min_final_train_origins),
        "--sku-universe-mode", args.sku_universe_mode,
        "--store-sku-sample-frac", str(args.store_sku_sample_frac),
        "--store-sku-sample-seed", str(args.store_sku_sample_seed),
    ]


def runtime_cache_args(args: argparse.Namespace) -> List[str]:
    out: List[str] = []
    if args.cache_dir:
        out += ["--cache-dir", args.cache_dir]
    if args.force_rebuild_cache:
        out += ["--force-rebuild-cache"]
    if args.quiet_progress:
        out += ["--quiet-progress"]
    return out


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    print("\n$", " ".join(shlex.quote(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def build_paths(work_dir: str) -> dict:
    root = Path(work_dir)
    return {
        "sanity": root / "00_sanity",
        "tuning": root / "01_tuning",
        "ablation": root / "02_ablation",
        "bench_base": root / "03_benchmark_baselines",
        "bench_prop": root / "04_benchmark_proposed",
        "stats": root / "05_stat_tests",
        "tables": root / "06_paper_tables",
    }


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace(" ", "_")


def tuning_dir(paths: dict, model_name: str) -> Path:
    return paths["tuning"] / model_slug(model_name)


def benchmark_model_dir(paths: dict, model_name: str) -> Path:
    return paths["bench_prop"] / model_slug(model_name)


def merge_csv_files(csv_paths: Iterable[Path], output_path: Path) -> Path:
    frames = []
    for p in csv_paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No CSV files found to merge into {output_path}")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def merge_json_files(json_paths: Iterable[Path], output_path: Path) -> Path:
    payload = {}
    for p in json_paths:
        p = Path(p)
        if p.exists():
            payload[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
    if not payload:
        raise FileNotFoundError(f"No JSON files found to merge into {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recommended M5 paper experiment workflow in the order: cache -> sanity -> tuning -> ablation -> benchmark -> stats -> tables.")
    parser.add_argument("--data-dir", required=True, help="Directory containing the raw M5 CSV files.")
    parser.add_argument("--work-dir", required=True, help="Directory for all intermediate and final outputs.")
    parser.add_argument("--cache-dir", default="", help="Optional shared cache directory.")
    parser.add_argument("--tasks", default="store_dept,store_cat,state_dept")
    parser.add_argument("--proposed-models", default="M3_FullSkuTemporalCNN,DeepSets,SetTransformer", help="Comma-separated proposed models to tune and benchmark end-to-end.")
    parser.add_argument("--ablation-params-model", default="M3_FullSkuTemporalCNN", help="Which tuned proposed model supplies params to the M0/M1/M3 ablation run.")
    parser.add_argument("--phases", default="cache,sanity,tune,ablation,benchmark,stats,tables", help="Comma-separated subset of phases to run. Allowed: cache,sanity,tune,ablation,benchmark,stats,tables,all")
    parser.add_argument("--mode", default="holdout", choices=["rolling", "holdout"], help="Main evaluation mode for ablation and final benchmark.")
    parser.add_argument("--sanity-task", default="store_dept")
    parser.add_argument("--quantiles", default="0.005,0.025,0.165,0.25,0.5,0.75,0.835,0.975,0.995")
    parser.add_argument("--t-hist", type=int, default=364, help="Historical lookback length. Use smaller values only for debugging.")
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--valid-size", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=4)
    parser.add_argument("--internal-valid-size", type=int, default=3)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--min-train-origins", type=int, default=20)
    parser.add_argument("--min-final-train-origins", type=int, default=20)
    parser.add_argument("--sku-universe-mode", default="history", choices=["history", "history_or_future"])
    parser.add_argument("--store-sku-sample-frac", type=float, default=1.0, help="Fraction of unique item-store units to keep globally across all tasks.")
    parser.add_argument("--store-sku-sample-seed", type=int, default=42, help="Random seed for reproducible global item-store sampling.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Pass through to phases that build/load M5 caches.")
    parser.add_argument("--quiet-progress", action="store_true", help="Reduce preprocessing progress output in child scripts.")
    parser.add_argument("--recent-window", type=int, default=14)
    parser.add_argument("--lag-windows", default="7,14,28,56")
    parser.add_argument("--stat-windows", default="7,28,56")
    parser.add_argument("--stage1-n-trials", type=int, default=96)
    parser.add_argument("--stage1-train-frac", type=float, default=0.25)
    parser.add_argument("--stage1-epochs", type=int, default=8)
    parser.add_argument("--stage2-top-k", type=int, default=12)
    parser.add_argument("--stage2-train-frac", type=float, default=0.6)
    parser.add_argument("--stage2-epochs", type=int, default=16)
    parser.add_argument("--final-epochs", type=int, default=24)
    parser.add_argument("--final-patience", type=int, default=6)
    parser.add_argument("--stats-metric", default="wspl", choices=["wrmsse", "wape", "wspl"], help="Metric used for Friedman/Nemenyi paper comparison. Default is WSPL so the paper stats align with quantile evaluation.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    phases = split_csv(args.phases)
    if "all" in phases:
        phases = ["cache", "sanity", "tune", "ablation", "benchmark", "stats", "tables"]

    paths = build_paths(args.work_dir)
    for p in paths.values():
        ensure_dir(str(p))

    split_args = split_control_args(args)
    cache_args = runtime_cache_args(args)

    proposed_models = split_csv(args.proposed_models)
    if not proposed_models:
        raise ValueError("--proposed-models must contain at least one model name.")
    ablation_params_model = args.ablation_params_model.strip() or "M3_FullSkuTemporalCNN"
    if ablation_params_model not in proposed_models:
        proposed_models = [ablation_params_model] + [m for m in proposed_models if m != ablation_params_model]

    best_params_json = str(tuning_dir(paths, ablation_params_model) / "best_params.json")
    benchmark_rows_csv = str(paths["bench_base"] / f"all_{args.mode}_rows.csv")
    benchmark_series_rows_csv = str(paths["bench_base"] / f"all_{args.mode}_series_rows.csv")
    workflow_tasks = split_csv(args.tasks)
    cache_tasks = list(dict.fromkeys(workflow_tasks + [args.sanity_task]))

    proposed_rows_csv = str(paths["bench_prop"] / f"proposed_all_{args.mode}_rows.csv")
    proposed_series_rows_csv = str(paths["bench_prop"] / f"proposed_all_{args.mode}_series_rows.csv")
    ablation_rows_csv = str(paths["ablation"] / f"all_ablation_rows_{args.mode}.csv")
    ablation_series_rows_csv = str(paths["ablation"] / f"all_ablation_series_rows_{args.mode}.csv")
    proposed_point_summary_csv = str(paths["bench_prop"] / f"proposed_point_summary_{args.mode}.csv")
    proposed_quantile_summary_csv = str(paths["bench_prop"] / f"proposed_quantile_summary_{args.mode}.csv")
    proposed_params_manifest_json = str(paths["tuning"] / "best_params_all_models.json")

    if "cache" in phases:
        cmd = [
            args.python, str(scripts_dir / "warm_m5_cache.py"),
            "--data-dir", args.data_dir,
            "--tasks", ",".join(cache_tasks),
        ] + split_args
        # Warm the main evaluation mode. If tuning/sanity phases are requested under rolling mode,
        # also warm holdout because those phases intentionally use holdout-style internal validation.
        if args.mode == "holdout":
            cmd += ["--holdout"]
        else:
            cmd += ["--rolling"]
            if any(p in phases for p in ["sanity", "tune"]):
                cmd += ["--holdout"]
        cmd += cache_args
        run_cmd(cmd, args.dry_run)

    if "sanity" in phases:
        cmd = [
            args.python, str(scripts_dir / "run_m5_proposed_models.py"),
            "--data-dir", args.data_dir,
            "--output-dir", str(paths["sanity"]),
            "--mode", "holdout",
            "--tasks", args.sanity_task,
            "--model-names", ",".join(proposed_models),
            "--epochs", "6",
            "--patience", "3",
            "--batch-size", "16",
            "--hidden-dim", "64",
            "--dropout", "0.10",
            "--cnn-num-blocks", "3",
            "--cnn-kernel-size", "3",
            "--quantiles", "",
            "--recent-window", str(args.recent_window),
            "--lag-windows", args.lag_windows,
            "--stat-windows", args.stat_windows,
        ] + split_args + cache_args
        run_cmd(cmd, args.dry_run)

    if "tune" in phases:
        tuned_json_paths = []
        for model_name in proposed_models:
            model_out = tuning_dir(paths, model_name)
            ensure_dir(str(model_out))
            cmd = [
                args.python, str(scripts_dir / "tune_m5_proposed_models.py"),
                "--data-dir", args.data_dir,
                "--output-dir", str(model_out),
                "--mode", "holdout",
                "--tasks", args.tasks,
                "--model-name", model_name,
                "--recent-window", str(args.recent_window),
                "--lag-windows", args.lag_windows,
                "--stat-windows", args.stat_windows,
                "--stage1-n-trials", str(args.stage1_n_trials),
                "--stage1-train-frac", str(args.stage1_train_frac),
                "--stage1-epochs", str(args.stage1_epochs),
                "--stage2-top-k", str(args.stage2_top_k),
                "--stage2-train-frac", str(args.stage2_train_frac),
                "--stage2-epochs", str(args.stage2_epochs),
                "--final-epochs", str(args.final_epochs),
                "--final-patience", str(args.final_patience),
            ] + split_args + cache_args
            run_cmd(cmd, args.dry_run)
            tuned_json_paths.append(model_out / "best_params.json")
        if not args.dry_run:
            merge_json_files(tuned_json_paths, Path(proposed_params_manifest_json))

    if "ablation" in phases:
        cmd = [
            args.python, str(scripts_dir / "run_m5_ablation.py"),
            "--data-dir", args.data_dir,
            "--output-dir", str(paths["ablation"]),
            "--mode", args.mode,
            "--tasks", args.tasks,
            "--recent-window", str(args.recent_window),
            "--lag-windows", args.lag_windows,
            "--stat-windows", args.stat_windows,
            "--params-json", best_params_json,
            "--quantiles", args.quantiles,
            "--epochs", str(args.final_epochs),
            "--patience", str(args.final_patience),
        ] + split_args + cache_args
        run_cmd(cmd, args.dry_run)

    if "benchmark" in phases:
        cmd_base = [
            args.python, str(scripts_dir / "run_m5_full_benchmark.py"),
            "--data-dir", args.data_dir,
            "--output-dir", str(paths["bench_base"]),
            "--mode", args.mode,
            "--tasks", args.tasks,
            "--quantiles", args.quantiles,
        ] + split_args + cache_args
        run_cmd(cmd_base, args.dry_run)

        proposed_row_paths = []
        proposed_series_row_paths = []
        proposed_point_paths = []
        proposed_quantile_paths = []
        for model_name in proposed_models:
            model_out = benchmark_model_dir(paths, model_name)
            ensure_dir(str(model_out))
            params_json = tuning_dir(paths, model_name) / "best_params.json"
            cmd_prop = [
                args.python, str(scripts_dir / "run_m5_proposed_models.py"),
                "--data-dir", args.data_dir,
                "--output-dir", str(model_out),
                "--mode", args.mode,
                "--tasks", args.tasks,
                "--quantiles", args.quantiles,
                "--model-names", model_name,
                "--recent-window", str(args.recent_window),
                "--lag-windows", args.lag_windows,
                "--stat-windows", args.stat_windows,
                "--params-json", str(params_json),
            ] + split_args + cache_args
            run_cmd(cmd_prop, args.dry_run)
            proposed_row_paths.append(model_out / f"proposed_all_{args.mode}_rows.csv")
            proposed_series_row_paths.append(model_out / f"proposed_all_{args.mode}_series_rows.csv")
            proposed_point_paths.append(model_out / f"proposed_point_summary_{args.mode}.csv")
            if args.quantiles:
                proposed_quantile_paths.append(model_out / f"proposed_quantile_summary_{args.mode}.csv")

        if not args.dry_run:
            merge_csv_files(proposed_row_paths, Path(proposed_rows_csv))
            merge_csv_files(proposed_series_row_paths, Path(proposed_series_rows_csv))
            merge_csv_files(proposed_point_paths, Path(proposed_point_summary_csv))
            if proposed_quantile_paths:
                merge_csv_files(proposed_quantile_paths, Path(proposed_quantile_summary_csv))

    if "stats" in phases:
        cmd = [
            args.python, str(scripts_dir / "run_m5_stat_tests.py"),
            "--benchmark-rows", benchmark_rows_csv,
            "--proposed-rows", proposed_rows_csv,
            "--ablation-rows", ablation_rows_csv,
            "--benchmark-series-rows", benchmark_series_rows_csv,
            "--proposed-series-rows", proposed_series_rows_csv,
            "--ablation-series-rows", ablation_series_rows_csv,
            "--ablation-models", "M0_AggHistOnly,M1_AggHistFutureSummary",
            "--metric", args.stats_metric,
            "--analysis-level", "series",
            "--series-block-cols", "task,agg_id",
            "--output-dir", str(paths["stats"]),
        ]
        run_cmd(cmd, args.dry_run)

    if "tables" in phases:
        cmd = [
            args.python, str(scripts_dir / "make_m5_paper_tables.py"),
            "--benchmark-rows", benchmark_rows_csv,
            "--proposed-rows", proposed_rows_csv,
            "--ablation-rows", ablation_rows_csv,
            "--output-dir", str(paths["tables"]),
        ]
        run_cmd(cmd, args.dry_run)

    print("\nWorkflow paths:")
    print(f"  proposed models:    {', '.join(proposed_models)}")
    print(f"  ablation params:    {best_params_json}")
    print(f"  tuned params all:   {proposed_params_manifest_json}")
    print(f"  benchmark rows:     {benchmark_rows_csv}")
    print(f"  benchmark series:   {benchmark_series_rows_csv}")
    print(f"  proposed rows:      {proposed_rows_csv}")
    print(f"  proposed series:    {proposed_series_rows_csv}")
    print(f"  ablation rows:      {ablation_rows_csv}")
    print(f"  ablation series:    {ablation_series_rows_csv}")
    print(f"  statistical tests:  {paths['stats']}")
    print(f"  final tables dir:   {paths['tables']}")


if __name__ == "__main__":
    main()
