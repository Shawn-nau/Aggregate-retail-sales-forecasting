from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch

from prepare_m5_experiments import (
    M5ExperimentConfig,
    build_m5_experiment_samples,
    default_cache_dir,
)
from compute_m5_metrics import prepare_aggregate_history_store
from m5_benchmarks import (
    build_benchmark_suite,
    run_point_benchmark,
    clone_benchmark,
)
from run_m5_proposed_models import (
    set_seed,
    configure_torch_runtime,
    FitConfig,
    TrainRegularizationConfig,
    run_one_setting,
)
from experiment_tracking import WandbRunWrapper, parse_tags, save_json


DEFAULT_SCARCITY_MODELS = ("AggregateHistGB", "M3_FullSkuTemporalCNN", "DeepSets")
DEFAULT_CAPACITY_GRID = (
    ("M3_h32", "M3_FullSkuTemporalCNN", 32, 4, 2),
    ("M3_h64", "M3_FullSkuTemporalCNN", 64, 4, 2),
    ("M3_h128", "M3_FullSkuTemporalCNN", 128, 4, 2),
    ("DeepSets_h32", "DeepSets", 32, 4, 2),
    ("DeepSets_h64", "DeepSets", 64, 4, 2),
    ("DeepSets_h128", "DeepSets", 128, 4, 2),
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_csv(text: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def subset_samples_by_origin_fraction(samples: List[Dict[str, Any]], frac: float) -> List[Dict[str, Any]]:
    origins = sorted({int(s["meta"]["origin_time"]) for s in samples})
    k = max(1, int(round(len(origins) * frac)))
    keep = set(origins[:k])
    return [s for s in samples if int(s["meta"]["origin_time"]) in keep]


def summarize_rows(df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    agg = {c: ["mean", "std"] for c in metric_cols if c in df.columns}
    out = df.groupby(list(group_cols), as_index=False).agg(agg)
    out.columns = ["_".join(col).strip("_") for col in out.columns.to_flat_index()]
    return out


def run_model(
    model_name: str,
    train_samples_raw: List[Dict[str, Any]],
    valid_samples_raw: List[Dict[str, Any]],
    hist_store: Dict[str, pd.DataFrame],
    task: str,
    device: torch.device,
    fit_cfg: FitConfig,
    reg_cfg: TrainRegularizationConfig,
    hidden_dim: int,
    dropout: float,
    num_heads: int,
    num_sab_layers: int,
    num_seeds: int,
    num_inducing_points: int,
    recent_window: int,
    lag_windows: Sequence[int],
    stat_windows: Sequence[int],
    monotone_quantiles: bool,
    cnn_num_blocks: int,
    cnn_kernel_size: int,
) -> Dict[str, Any]:
    return run_one_setting(
        model_name=model_name,
        train_samples_raw=train_samples_raw,
        valid_samples_raw=valid_samples_raw,
        test_like_samples_raw=valid_samples_raw,
        agg_history_store=hist_store,
        task=task,
        device=device,
        fit_cfg=fit_cfg,
        reg_cfg=reg_cfg,
        quantiles=None,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_heads=num_heads,
        num_sab_layers=num_sab_layers,
        num_seeds=num_seeds,
        num_inducing_points=num_inducing_points,
        recent_window=recent_window,
        lag_windows=lag_windows,
        stat_windows=stat_windows,
        monotone_quantiles=monotone_quantiles,
        cnn_num_blocks=cnn_num_blocks,
        cnn_kernel_size=cnn_kernel_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling robustness experiments on M5 tasks.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./m5_robustness_outputs")
    parser.add_argument("--tasks", type=str, default="store_dept")
    parser.add_argument("--scarcity-fracs", type=str, default="1.0,0.75,0.5,0.25")
    parser.add_argument("--scarcity-models", type=str, default=",".join(DEFAULT_SCARCITY_MODELS))
    parser.add_argument("--include-set-transformer-capacity", action="store_true")
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
    parser.add_argument("--store-sku-sample-frac", type=float, default=1.0)
    parser.add_argument("--store-sku-sample-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-sab-layers", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-inducing-points", type=int, default=32)
    parser.add_argument("--recent-window", type=int, default=14)
    parser.add_argument("--lag-windows", type=str, default="7,14,28,56")
    parser.add_argument("--stat-windows", type=str, default="7,28,56")
    parser.add_argument("--cnn-num-blocks", type=int, default=3)
    parser.add_argument("--cnn-kernel-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="m5-cross-level-forecasting")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="robustness")
    parser.add_argument("--wandb-tags", type=str, default="m5,robustness")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--save-run-config", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch_runtime(device, cpu_threads=args.cpu_threads)

    tasks = parse_csv(args.tasks)
    scarcity_fracs = tuple(float(x) for x in parse_csv(args.scarcity_fracs))
    scarcity_models = parse_csv(args.scarcity_models)
    lag_windows = tuple(int(x) for x in parse_csv(args.lag_windows))
    stat_windows = tuple(int(x) for x in parse_csv(args.stat_windows))

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
    reg_cfg = TrainRegularizationConfig(
        use_permutation_consistency=True,
        permutation_prob=0.5,
        consistency_weight=0.01,
    )

    cache_dir = args.cache_dir or default_cache_dir(cfg)
    print(f"Using cache directory: {cache_dir}")
    experiment = build_m5_experiment_samples(
        raw_data=None,
        cfg=cfg,
        need_outer=False,
        need_holdout=False,
        need_rolling=True,
        cache_dir=cache_dir,
        force_rebuild=args.force_rebuild_cache,
        progress=(not args.quiet_progress),
    )

    ensure_dir(args.output_dir)
    run_config = {
        "cfg": asdict(cfg),
        "fit_cfg": asdict(fit_cfg),
        "reg_cfg": asdict(reg_cfg),
        "cache_dir": cache_dir,
        "scarcity_fracs": list(scarcity_fracs),
        "scarcity_models": list(scarcity_models),
        "seed": args.seed,
    }
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
        job_type="robustness",
    )

    bench_suite = build_benchmark_suite(random_state=args.seed)
    capacity_grid = list(DEFAULT_CAPACITY_GRID)
    if args.include_set_transformer_capacity:
        capacity_grid.extend([
            ("SetTransformer_h64_L1", "SetTransformer", 64, 4, 1),
            ("SetTransformer_h64_L2", "SetTransformer", 64, 4, 2),
        ])

    scarcity_rows: List[Dict[str, Any]] = []
    capacity_rows: List[Dict[str, Any]] = []

    for task in tasks:
        print(f"\n=== Rolling robustness task: {task} ===")
        agg_daily = experiment[task]["task_package"]["agg_daily_df"]
        hist_store = prepare_aggregate_history_store(agg_daily)

        for fold_obj in experiment[task]["rolling_fold_samples"]:
            fold_id = int(fold_obj["fold_id"])
            train_samples_raw = fold_obj["train_samples"]
            valid_samples_raw = fold_obj["valid_samples"]

            for frac in scarcity_fracs:
                train_sub = subset_samples_by_origin_fraction(train_samples_raw, frac)
                frac_label = f"{int(round(frac * 100))}%"
                for model_name in scarcity_models:
                    print(f"[Robustness-Scarcity] Task={task} Fold={fold_id} Frac={frac_label} Model={model_name}")
                    if model_name == "AggregateHistGB":
                        out = run_point_benchmark(
                            benchmark=clone_benchmark(bench_suite["AggregateHistGB"]),
                            train_samples=train_sub,
                            eval_samples=valid_samples_raw,
                            agg_history_store=hist_store,
                            task=task,
                            horizon=valid_samples_raw[0]["y_target"].shape[0],
                        )
                    else:
                        out = run_model(
                            model_name=model_name,
                            train_samples_raw=train_sub,
                            valid_samples_raw=valid_samples_raw,
                            hist_store=hist_store,
                            task=task,
                            device=device,
                            fit_cfg=fit_cfg,
                            reg_cfg=reg_cfg,
                            hidden_dim=args.hidden_dim,
                            dropout=args.dropout,
                            num_heads=args.num_heads,
                            num_sab_layers=args.num_sab_layers,
                            num_seeds=args.num_seeds,
                            num_inducing_points=args.num_inducing_points,
                            recent_window=args.recent_window,
                            lag_windows=lag_windows,
                            stat_windows=stat_windows,
                            monotone_quantiles=True,
                            cnn_num_blocks=args.cnn_num_blocks,
                            cnn_kernel_size=args.cnn_kernel_size,
                        )
                    scarcity_rows.append({
                        "task": task,
                        "fold": fold_id,
                        "training_origins_used": frac_label,
                        "model": model_name,
                        "wrmsse": float(out["wrmsse"]),
                        "wape": float(out["wape"]),
                    })

            for label, base_model, hidden_dim, num_heads, num_sab_layers in capacity_grid:
                print(f"[Robustness-Capacity] Task={task} Fold={fold_id} Model={label}")
                out = run_model(
                    model_name=base_model,
                    train_samples_raw=train_samples_raw,
                    valid_samples_raw=valid_samples_raw,
                    hist_store=hist_store,
                    task=task,
                    device=device,
                    fit_cfg=fit_cfg,
                    reg_cfg=reg_cfg,
                    hidden_dim=hidden_dim,
                    dropout=args.dropout,
                    num_heads=num_heads,
                    num_sab_layers=num_sab_layers,
                    num_seeds=args.num_seeds,
                    num_inducing_points=args.num_inducing_points,
                    recent_window=args.recent_window,
                    lag_windows=lag_windows,
                    stat_windows=stat_windows,
                    monotone_quantiles=True,
                    cnn_num_blocks=args.cnn_num_blocks,
                    cnn_kernel_size=args.cnn_kernel_size,
                )
                hist = out["history"]
                train_final = float(hist["train_loss"][-1]) if hist["train_loss"] else float("nan")
                valid_final = float(hist["valid_loss"][-1]) if hist["valid_loss"] else float("nan")
                overfit_gap = valid_final - train_final if pd.notna(train_final) and pd.notna(valid_final) else float("nan")
                capacity_rows.append({
                    "task": task,
                    "fold": fold_id,
                    "model_capacity": label,
                    "base_model": base_model,
                    "hidden_dim": hidden_dim,
                    "num_heads": num_heads,
                    "num_sab_layers": num_sab_layers,
                    "wrmsse": float(out["wrmsse"]),
                    "wape": float(out["wape"]),
                    "train_final_loss": train_final,
                    "valid_final_loss": valid_final,
                    "overfitting_gap": overfit_gap,
                })

    scarcity_df = pd.DataFrame(scarcity_rows)
    scarcity_df.to_csv(os.path.join(args.output_dir, "Table5A_rolling_data_scarcity_rows.csv"), index=False)
    scarcity_summary = summarize_rows(scarcity_df, ["task", "training_origins_used", "model"], ["wrmsse", "wape"])
    scarcity_summary.to_csv(os.path.join(args.output_dir, "Table5A_rolling_data_scarcity_summary.csv"), index=False)

    capacity_df = pd.DataFrame(capacity_rows)
    capacity_df.to_csv(os.path.join(args.output_dir, "Table5B_rolling_capacity_rows.csv"), index=False)
    capacity_summary = summarize_rows(
        capacity_df,
        ["task", "model_capacity", "base_model", "hidden_dim", "num_heads", "num_sab_layers"],
        ["wrmsse", "wape", "train_final_loss", "valid_final_loss", "overfitting_gap"],
    )
    capacity_summary.to_csv(os.path.join(args.output_dir, "Table5B_rolling_capacity_summary.csv"), index=False)

    print("\nTable 5A rolling scarcity summary:")
    print(scarcity_summary)
    print("\nTable 5B rolling capacity summary:")
    print(capacity_summary)

    if wandb_run.active:
        wandb_run.log_dataframe("robustness_scarcity_rows", scarcity_df)
        wandb_run.log_dataframe("robustness_capacity_rows", capacity_df)
        wandb_run.finish()

    print(f"\nDone. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
