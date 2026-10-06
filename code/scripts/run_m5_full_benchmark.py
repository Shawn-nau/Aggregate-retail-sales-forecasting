from __future__ import annotations

import argparse
import os
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List

import pandas as pd

from prepare_m5_experiments import (
    M5ExperimentConfig,
    build_m5_experiment_samples,
    default_cache_dir,
)
from compute_m5_metrics import (
    build_task_aggregate_daily,
    prepare_aggregate_history_store,
    summarize_point_results,
    summarize_quantile_results,
    build_main_point_table,
    build_main_quantile_table,
)
from m5_benchmarks import (
    build_benchmark_suite,
    run_benchmark_suite_on_task,
)
from experiment_tracking import WandbRunWrapper, parse_tags, save_json


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_df(df: pd.DataFrame, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    df.to_csv(path, index=True if df.index.name is not None else False)


def main(args: argparse.Namespace) -> None:
    tasks = tuple([t.strip() for t in args.tasks.split(",") if t.strip()])
    quantiles = [float(x) for x in args.quantiles.split(",")] if args.quantiles else None

    cfg = M5ExperimentConfig(
        data_dir=args.data_dir,
        t_hist=args.t_hist,
        horizon=args.horizon,
        tasks=tasks,
        valid_size=args.valid_size,
        test_size=args.test_size,
        internal_valid_size=args.internal_valid_size,
        gap=args.gap,
        min_train_origins=args.min_train_origins,
        min_final_train_origins=args.min_final_train_origins,
        fill_value=args.fill_value,
        sku_universe_mode=args.sku_universe_mode,
        store_sku_sample_frac=args.store_sku_sample_frac,
        store_sku_sample_seed=args.store_sku_sample_seed,
    )

    cache_dir = args.cache_dir or default_cache_dir(cfg)
    print(f"Using cache directory: {cache_dir}")
    print("Building/loading cached experiment artifacts...")
    experiment = build_m5_experiment_samples(
        raw_data=None,
        cfg=cfg,
        need_outer=False,
        need_holdout=(args.mode == "holdout"),
        need_rolling=(args.mode == "rolling"),
        cache_dir=cache_dir,
        force_rebuild=args.force_rebuild_cache,
        progress=(not args.quiet_progress),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    run_config = {
        "cfg": asdict(cfg),
        "mode": args.mode,
        "quantiles": quantiles,
        "cache_dir": cache_dir,
        "random_state": args.random_state,
    }
    if args.save_run_config:
        save_json(run_config, os.path.join(args.output_dir, f"benchmark_run_config_{args.mode}.json"))
    wandb_run = WandbRunWrapper(
        enabled=(args.use_wandb and args.wandb_mode != "disabled"),
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=parse_tags(args.wandb_tags),
        config=run_config,
        mode=args.wandb_mode,
        job_type="benchmarks",
    )

    print("Building benchmark suite...")
    suite = build_benchmark_suite(random_state=args.random_state)

    model_filter = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if model_filter:
        unknown = [m for m in model_filter if m not in suite]
        if unknown:
            raise ValueError(f"Unknown benchmark model(s): {unknown}. Available: {sorted(suite)}")
        suite = {m: suite[m] for m in model_filter}
        print(f"Running selected benchmark models: {model_filter}")

    all_rows: List[pd.DataFrame] = []
    all_series_rows: List[pd.DataFrame] = []

    for task in cfg.tasks:
        print(f"\n=== Running task: {task} ===")
        agg_daily = experiment[task]["task_package"]["agg_daily_df"]
        hist_store = prepare_aggregate_history_store(agg_daily)

        task_rows, task_series_rows = run_benchmark_suite_on_task(
            benchmark_suite=suite,
            task_name=task,
            task_obj=experiment[task],
            agg_history_store=hist_store,
            quantiles=quantiles,
            mode=args.mode,
            return_series_rows=True,
        )
        all_rows.append(task_rows)
        if task_series_rows is not None and not task_series_rows.empty:
            all_series_rows.append(task_series_rows)

        task_dir = os.path.join(args.output_dir, task)
        ensure_dir(task_dir)
        task_rows.to_csv(os.path.join(task_dir, f"{args.mode}_rows.csv"), index=False)
        if task_series_rows is not None and not task_series_rows.empty:
            task_series_rows.to_csv(os.path.join(task_dir, f"{args.mode}_series_rows.csv"), index=False)
        if wandb_run.active:
            wandb_run.log_dataframe(f"{task}_{args.mode}_rows", task_rows)

    all_rows_df = pd.concat(all_rows, axis=0, ignore_index=True)
    ensure_dir(args.output_dir)
    all_rows_df.to_csv(os.path.join(args.output_dir, f"all_{args.mode}_rows.csv"), index=False)
    if all_series_rows:
        all_series_rows_df = pd.concat(all_series_rows, axis=0, ignore_index=True)
        all_series_rows_df.to_csv(os.path.join(args.output_dir, f"all_{args.mode}_series_rows.csv"), index=False)
    else:
        all_series_rows_df = pd.DataFrame()

    point_summary = summarize_point_results(all_rows_df.to_dict("records"))
    point_summary.to_csv(os.path.join(args.output_dir, f"point_summary_{args.mode}.csv"), index=False)

    point_table = build_main_point_table(point_summary)
    point_table.to_csv(os.path.join(args.output_dir, f"Table2_main_point_{args.mode}.csv"))

    print("\nPoint summary:")
    print(point_summary)
    print("\nMain point table:")
    print(point_table)

    if quantiles is not None:
        quant_summary = summarize_quantile_results(all_rows_df.to_dict("records"))
        quant_summary.to_csv(os.path.join(args.output_dir, f"quantile_summary_{args.mode}.csv"), index=False)

        quant_table = build_main_quantile_table(quant_summary)
        quant_table.to_csv(os.path.join(args.output_dir, f"Table3_main_quantile_{args.mode}.csv"))

        print("\nQuantile summary:")
        print(quant_summary)
        print("\nMain quantile table:")
        print(quant_table)

    print(f"\nDone. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full M5 benchmark suite and export paper-ready CSV tables.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing M5 raw CSV files.")
    parser.add_argument("--output-dir", type=str, default="./m5_outputs", help="Directory to save CSV outputs.")
    parser.add_argument("--mode", type=str, default="rolling", choices=["rolling", "holdout"], help="Evaluation mode.")
    parser.add_argument("--tasks", type=str, default="store_dept,store_cat,state_dept", help="Comma-separated task names.")
    parser.add_argument("--models", type=str, default="", help="Comma-separated benchmark model names to run. Empty string runs the full suite.")
    parser.add_argument("--quantiles", type=str, default="0.005,0.025,0.165,0.25,0.5,0.75,0.835,0.975,0.995", help="Comma-separated quantile levels. Pass empty string to skip quantile benchmarks.")
    parser.add_argument("--t-hist", type=int, default=364)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--valid-size", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=4)
    parser.add_argument("--internal-valid-size", type=int, default=3)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--min-train-origins", type=int, default=20)
    parser.add_argument("--min-final-train-origins", type=int, default=20)
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument("--sku-universe-mode", type=str, default="history", choices=["history", "history_or_future"])
    parser.add_argument("--store-sku-sample-frac", type=float, default=1.0, help="Fraction of unique item-store units to keep globally across all tasks.")
    parser.add_argument("--store-sku-sample-seed", type=int, default=42, help="Random seed for reproducible global item-store sampling.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true", help="Log experiment summaries to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="m5-cross-level-forecasting")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="benchmarks")
    parser.add_argument("--wandb-tags", type=str, default="m5,benchmarks")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--save-run-config", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="", help="Directory for reusable preprocessing caches.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild cached preprocessing artifacts.")
    parser.add_argument("--quiet-progress", action="store_true", help="Reduce preprocessing progress/ETA logging.")
    args = parser.parse_args()
    main(args)