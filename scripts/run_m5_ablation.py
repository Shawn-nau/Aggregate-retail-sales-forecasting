from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, Any, List, Sequence

import pandas as pd
import torch

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
from run_m5_proposed_models import (
    set_seed,
    configure_torch_runtime,
    FitConfig,
    TrainRegularizationConfig,
    run_one_setting,
    apply_overrides_from_params,
)
from experiment_tracking import WandbRunWrapper, parse_tags, save_json


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_tasks(s: str):
    return tuple([x.strip() for x in s.split(",") if x.strip()])


def run_neural_ablation_rows(
    task_name: str,
    task_obj: Dict[str, Any],
    agg_history_store: Dict[str, pd.DataFrame],
    device: torch.device,
    fit_cfg: FitConfig,
    hidden_dim: int,
    dropout: float,
    num_heads: int,
    num_sab_layers: int,
    num_seeds: int,
    mode: str,
    recent_window: int,
    lag_windows: Sequence[int],
    stat_windows: Sequence[int],
    monotone_quantiles: bool,
    cnn_num_blocks: int,
    cnn_kernel_size: int,
    quantiles: Sequence[float] | None = None,
    return_series_rows: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run neural ablation variants.

    Point rows are always produced so M0/M1 can participate in WRMSSE/rmsse
    statistical tests. If quantiles are supplied, a second quantile fit is also
    run for each variant so M0/M1 can participate in WSPL/mean_spl tests.
    """
    rows: List[Dict[str, Any]] = []
    series_rows: List[Dict[str, Any]] = []

    reg_off = TrainRegularizationConfig(
        use_permutation_consistency=False,
        permutation_prob=0.0,
        consistency_weight=0.0,
    )

    variants = [
        ("M0_AggHistOnly", "AggHistOnly", reg_off),
        ("M1_AggHistFutureSummary", "AggHistFutureSummary", reg_off),
        ("M3_FullSkuTemporalCNN", "SkuTemporalCNN", reg_off),
    ]

    def append_out(out: Dict[str, Any], label: str, fold_id: int, metric_kind: str) -> None:
        variant_series_rows = out.pop("series_rows", [])
        for rec in variant_series_rows:
            rec["task"] = task_name
            rec["model"] = label
            rec["base_model"] = label
            rec["fold"] = fold_id
        series_rows.extend(variant_series_rows)

        row = {
            "task": task_name,
            "model": label,
            "base_model": label,
            "fold": fold_id,
            "wape": out.get("wape", float("nan")),
        }
        if metric_kind == "point":
            row["wrmsse"] = out["wrmsse"]
        elif metric_kind == "quantile":
            row["wspl"] = out["wspl"]
        else:
            raise ValueError(f"Unknown metric_kind: {metric_kind}")
        rows.append(row)

    if mode == "rolling":
        fold_iter = [
            (
                int(fold_obj["fold_id"]),
                fold_obj["train_samples"],
                fold_obj["valid_samples"],
                fold_obj["valid_samples"],
            )
            for fold_obj in task_obj["rolling_fold_samples"]
        ]
    elif mode == "holdout":
        fold_iter = [
            (
                0,
                task_obj["final_train_samples"],
                task_obj["internal_valid_samples"],
                task_obj["test_samples"],
            )
        ]
    else:
        raise ValueError("mode must be one of {'rolling','holdout'}")

    for fold_id, train_samples_raw, valid_samples_raw, eval_samples_raw in fold_iter:
        for label, base_model, reg_cfg in variants:
            fold_text = f"Fold={fold_id}" if mode == "rolling" else "Holdout"
            print(f"[Ablation] Task={task_name} {fold_text} Model={label} Metric=point")
            point_out = run_one_setting(
                model_name=base_model,
                train_samples_raw=train_samples_raw,
                valid_samples_raw=valid_samples_raw,
                test_like_samples_raw=eval_samples_raw,
                agg_history_store=agg_history_store,
                task=task_name,
                device=device,
                fit_cfg=fit_cfg,
                reg_cfg=reg_cfg,
                quantiles=None,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_heads=num_heads,
                num_sab_layers=num_sab_layers,
                num_seeds=num_seeds,
                num_inducing_points=32,
                recent_window=recent_window,
                lag_windows=lag_windows,
                stat_windows=stat_windows,
                monotone_quantiles=monotone_quantiles,
                cnn_num_blocks=cnn_num_blocks,
                cnn_kernel_size=cnn_kernel_size,
            )
            append_out(point_out, label=label, fold_id=fold_id, metric_kind="point")

            if quantiles is not None:
                print(f"[Ablation] Task={task_name} {fold_text} Model={label} Metric=quantile")
                q_out = run_one_setting(
                    model_name=base_model + "_Q",
                    train_samples_raw=train_samples_raw,
                    valid_samples_raw=valid_samples_raw,
                    test_like_samples_raw=eval_samples_raw,
                    agg_history_store=agg_history_store,
                    task=task_name,
                    device=device,
                    fit_cfg=fit_cfg,
                    reg_cfg=reg_cfg,
                    quantiles=quantiles,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    num_heads=num_heads,
                    num_sab_layers=num_sab_layers,
                    num_seeds=num_seeds,
                    num_inducing_points=32,
                    recent_window=recent_window,
                    lag_windows=lag_windows,
                    stat_windows=stat_windows,
                    monotone_quantiles=monotone_quantiles,
                    cnn_num_blocks=cnn_num_blocks,
                    cnn_kernel_size=cnn_kernel_size,
                )
                append_out(q_out, label=label, fold_id=fold_id, metric_kind="quantile")

    rows_df = pd.DataFrame(rows)
    series_df = pd.DataFrame(series_rows)
    if return_series_rows:
        return rows_df, series_df
    return rows_df

def main():
    parser = argparse.ArgumentParser(description="Run updated SKU-information ablation experiments on M5 tasks.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./m5_ablation_outputs")
    parser.add_argument("--mode", type=str, default="rolling", choices=["rolling", "holdout"])
    parser.add_argument("--tasks", type=str, default="store_dept,store_cat,state_dept")
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
    parser.add_argument("--recent-window", type=int, default=14)
    parser.add_argument("--quantiles", type=str, default="0.005,0.025,0.165,0.25,0.5,0.75,0.835,0.975,0.995", help="Comma-separated quantile levels for quantile-based ablation. Pass empty string to fall back to point-metric ablation.")
    parser.add_argument("--lag-windows", type=str, default="7,14,28,56")
    parser.add_argument("--stat-windows", type=str, default="7,28,56")
    parser.add_argument("--disable-monotone-quantiles", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cnn-num-blocks", type=int, default=3, help="Number of temporal residual blocks in the SKU future encoder.")
    parser.add_argument("--cnn-kernel-size", type=int, default=3, choices=[3,5,7], help="Kernel size in the SKU future temporal CNN.")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-sab-layers", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params-json", type=str, default="", help="Optional JSON file of tuned hyperparameters to override CLI defaults.")
    parser.add_argument("--use-wandb", action="store_true", help="Log experiment summaries to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="m5-cross-level-forecasting")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="ablation")
    parser.add_argument("--wandb-tags", type=str, default="m5,ablation")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--save-run-config", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="", help="Directory for reusable preprocessing caches.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild cached preprocessing artifacts.")
    parser.add_argument("--quiet-progress", action="store_true", help="Reduce preprocessing progress/ETA logging.")
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU intra/inter-op thread limit for stable CPU training.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch_runtime(device, cpu_threads=args.cpu_threads)
    tasks = parse_tasks(args.tasks)

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
    fit_cfg = FitConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    reg_cfg_for_override = TrainRegularizationConfig(
        use_permutation_consistency=False,
        permutation_prob=0.0,
        consistency_weight=0.0,
    )
    apply_overrides_from_params(args, fit_cfg, reg_cfg_for_override)
    lag_windows = tuple(int(x) for x in args.lag_windows.split(",") if x.strip())
    stat_windows = tuple(int(x) for x in args.stat_windows.split(",") if x.strip())
    quantiles = [float(x) for x in args.quantiles.split(",") if x.strip()] if args.quantiles.strip() else None
    monotone_quantiles = not args.disable_monotone_quantiles

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
        "fit_cfg": asdict(fit_cfg),
        "mode": getattr(args, "mode", "holdout_only"),
        "cache_dir": cache_dir,
        "seed": args.seed,
        "recent_window": args.recent_window,
        "lag_windows": list(lag_windows),
        "stat_windows": list(stat_windows),
        "quantiles": quantiles,
        "monotone_quantiles": monotone_quantiles,
        "cnn_num_blocks": args.cnn_num_blocks,
        "cnn_kernel_size": args.cnn_kernel_size,
        "params_json": args.params_json,
    }
    if 'reg_cfg' in locals():
        run_config["reg_cfg"] = asdict(reg_cfg)
    if args.save_run_config:
        save_json(run_config, os.path.join(args.output_dir, "run_config.json"))
    wandb_run = WandbRunWrapper(
        enabled=(args.use_wandb and args.wandb_mode != "disabled"),
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=parse_tags(args.wandb_tags),
        config=run_config,
        mode=args.wandb_mode,
        job_type="ablation",
    )

    bench_suite_full = build_benchmark_suite(random_state=args.seed)
    bench_suite = {
        "AggregateHistGB": bench_suite_full["AggregateHistGB"],
        "ChildSummaryHistGB": bench_suite_full["ChildSummaryHistGB"],
        "BottomUpGlobalHistGB": bench_suite_full["BottomUpGlobalHistGB"],
    }

    all_rows = []
    all_series_rows = []
    for task in tasks:
        print(f"\n=== Ablation task: {task} ===")
        agg_daily = experiment[task]["task_package"]["agg_daily_df"]
        hist_store = prepare_aggregate_history_store(agg_daily)

        bench_rows, bench_series_rows = run_benchmark_suite_on_task(
            benchmark_suite=bench_suite,
            task_name=task,
            task_obj=experiment[task],
            agg_history_store=hist_store,
            quantiles=quantiles,
            mode=args.mode,
            return_series_rows=True,
        )

        neural_rows, neural_series_rows = run_neural_ablation_rows(
            task_name=task,
            task_obj=experiment[task],
            agg_history_store=hist_store,
            device=device,
            fit_cfg=fit_cfg,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_heads=args.num_heads,
            num_sab_layers=args.num_sab_layers,
            num_seeds=args.num_seeds,
            mode=args.mode,
            recent_window=args.recent_window,
            lag_windows=lag_windows,
            stat_windows=stat_windows,
            monotone_quantiles=monotone_quantiles,
            cnn_num_blocks=args.cnn_num_blocks,
            cnn_kernel_size=args.cnn_kernel_size,
            quantiles=quantiles,
            return_series_rows=True,
        )

        task_rows = pd.concat([bench_rows, neural_rows], axis=0, ignore_index=True)
        task_series_rows = pd.concat([bench_series_rows, neural_series_rows], axis=0, ignore_index=True)
        all_rows.append(task_rows)
        if not task_series_rows.empty:
            all_series_rows.append(task_series_rows)

        task_dir = os.path.join(args.output_dir, task)
        ensure_dir(task_dir)
        task_rows.to_csv(os.path.join(task_dir, f"ablation_rows_{args.mode}.csv"), index=False)
        if not task_series_rows.empty:
            task_series_rows.to_csv(os.path.join(task_dir, f"ablation_series_rows_{args.mode}.csv"), index=False)

    all_rows_df = pd.concat(all_rows, axis=0, ignore_index=True)
    ensure_dir(args.output_dir)
    all_rows_df.to_csv(os.path.join(args.output_dir, f"all_ablation_rows_{args.mode}.csv"), index=False)
    if all_series_rows:
        all_series_rows_df = pd.concat(all_series_rows, axis=0, ignore_index=True)
        all_series_rows_df.to_csv(os.path.join(args.output_dir, f"all_ablation_series_rows_{args.mode}.csv"), index=False)
    else:
        all_series_rows_df = pd.DataFrame()

    if quantiles is None:
        summary = summarize_point_results(all_rows_df.to_dict("records"))
        main_table = build_main_point_table(summary, metric_col="wrmsse_mean")
        metric_name = "wrmsse"
    else:
        summary = summarize_quantile_results(all_rows_df.to_dict("records"))
        main_table = build_main_quantile_table(summary, metric_col="wspl_mean")
        metric_name = "wspl"

    summary.to_csv(os.path.join(args.output_dir, f"ablation_sku_info_summary_{metric_name}_{args.mode}.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, f"ablation_sku_info_summary_{args.mode}.csv"), index=False)

    main_table.to_csv(os.path.join(args.output_dir, f"Table_ablation_sku_info_{metric_name}_{args.mode}.csv"))
    main_table.to_csv(os.path.join(args.output_dir, f"Table_ablation_sku_info_{args.mode}.csv"))

    print(f"\nAblation summary ({metric_name.upper()}):")
    print(summary)
    print(f"\nTable 4 ablation ({metric_name.upper()}):")
    print(main_table)
    print(f"\nDone. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()