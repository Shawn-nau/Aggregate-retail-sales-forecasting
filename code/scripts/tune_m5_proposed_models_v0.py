from __future__ import annotations

import argparse
import gc
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from prepare_m5_experiments import M5ExperimentConfig, build_m5_experiment_samples, default_cache_dir
from compute_m5_metrics import prepare_aggregate_history_store
from experiment_tracking import WandbRunWrapper, maybe_import_optuna, parse_tags, save_json
from run_m5_proposed_models import FitConfig, TrainRegularizationConfig, run_one_setting, set_seed, configure_torch_runtime


CANONICAL_MODEL_MAP = {
    "M0_AggHistOnly": "AggHistOnly",
    "M1_AggHistFutureSummary": "AggHistFutureSummary",
    "M3_FullSkuTemporalCNN": "SkuTemporalCNN",
    "AggHistOnly": "AggHistOnly",
    "AggHistFutureSummary": "AggHistFutureSummary",
    "SkuTemporalCNN": "SkuTemporalCNN",
    "DeepSets": "DeepSets",
    "SetTransformer": "SetTransformer",
    "M0": "AggHistOnly",
    "M1": "AggHistFutureSummary",
    "M3": "SkuTemporalCNN",
}


@dataclass
class StageBudget:
    name: str
    tasks: Tuple[str, ...]
    train_frac: float
    valid_frac: float
    eval_frac: float
    epochs: int
    patience: int
    max_folds: int


def parse_tasks(s: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def canonical_model_name(name: str) -> str:
    key = (name or "").strip()
    if key not in CANONICAL_MODEL_MAP:
        allowed = ", ".join(sorted(CANONICAL_MODEL_MAP))
        raise ValueError(f"Unknown model name '{name}'. Allowed aliases: {allowed}")
    return CANONICAL_MODEL_MAP[key]


def recent_subset(samples: List[Dict[str, Any]], frac: float, min_keep: int = 8) -> List[Dict[str, Any]]:
    if frac >= 0.999:
        return list(samples)
    n = len(samples)
    if n <= min_keep:
        return list(samples)
    k = max(min_keep, int(math.ceil(n * max(frac, 0.0))))
    k = min(k, n)
    return list(samples[-k:])


def select_fold_objects(task_obj: Dict[str, Any], mode: str, max_folds: int) -> List[Dict[str, Any]]:
    if mode == "rolling":
        folds = list(task_obj["rolling_fold_samples"])
        return folds[:max_folds] if max_folds > 0 else folds
    if mode == "holdout":
        return [{
            "fold_id": 0,
            "train_samples": task_obj["final_train_samples"],
            "valid_samples": task_obj["internal_valid_samples"],
            "test_samples": task_obj["test_samples"],
        }]
    raise ValueError("mode must be one of {'rolling','holdout'}")


def build_stage_fit_cfg(base_fit_cfg: FitConfig, stage: StageBudget, params: Dict[str, Any]) -> FitConfig:
    return FitConfig(
        epochs=stage.epochs,
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        batch_size=int(params["batch_size"]),
        patience=stage.patience,
        num_workers=base_fit_cfg.num_workers,
        max_grad_norm=base_fit_cfg.max_grad_norm,
        use_amp=base_fit_cfg.use_amp,
        scheduler_patience=min(base_fit_cfg.scheduler_patience, max(1, stage.patience // 2)),
        scheduler_factor=base_fit_cfg.scheduler_factor,
        print_every=max(base_fit_cfg.print_every, 999999),
    )


def suggest_params(trial: Any, fixed_model_name: str) -> Dict[str, Any]:
    chosen_model = canonical_model_name(fixed_model_name) if fixed_model_name else canonical_model_name(
        trial.suggest_categorical("model_name", ["M0_AggHistOnly", "M1_AggHistFutureSummary", "M3_FullSkuTemporalCNN", "DeepSets", "SetTransformer"])
    )
    params: Dict[str, Any] = {
        "model_name": chosen_model,
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 96, 128, 160]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.25),
        "lr": trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16,32,64]),
        "num_heads": 4,
        "num_sab_layers": 2,
        "num_seeds": 1,
        "num_inducing_points": 32,
        "cnn_num_blocks": 3,
        "cnn_kernel_size": 3,
    }
    if chosen_model in {"SkuTemporalCNN", "DeepSets", "SetTransformer"}:
        params["cnn_num_blocks"] = trial.suggest_categorical("cnn_num_blocks", [2, 3, 4])
        params["cnn_kernel_size"] = trial.suggest_categorical("cnn_kernel_size", [3, 5])
    if chosen_model == "SetTransformer":
        params["num_heads"] = trial.suggest_categorical("num_heads", [2, 4, 8])
        params["num_sab_layers"] = trial.suggest_categorical("num_sab_layers", [1, 2, 3])
        params["num_seeds"] = trial.suggest_categorical("num_seeds", [1, 2])
        params["num_inducing_points"] = trial.suggest_categorical("num_inducing_points", [16, 32, 64])
    return params


def objective_rows_mean(rows: List[Dict[str, Any]], quantiles: Optional[Sequence[float]]) -> float:
    key = "wspl" if quantiles is not None else "wrmsse"
    vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("inf")


def evaluate_config_on_stage(
    *,
    experiment: Dict[str, Any],
    stage: StageBudget,
    mode: str,
    quantiles: Optional[Sequence[float]],
    device: torch.device,
    base_fit_cfg: FitConfig,
    reg_cfg: TrainRegularizationConfig,
    params: Dict[str, Any],
    recent_window: int,
    lag_windows: Sequence[int],
    stat_windows: Sequence[int],
    monotone_quantiles: bool,
    use_test_for_holdout: bool,
    trial: Any = None,
    wandb_run: Optional[WandbRunWrapper] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    fit_cfg = build_stage_fit_cfg(base_fit_cfg, stage, params)
    all_rows: List[Dict[str, Any]] = []
    step_count = 0
    for task_name in stage.tasks:
        task_obj = experiment[task_name]
        agg_daily = task_obj["task_package"]["agg_daily_df"]
        hist_store = prepare_aggregate_history_store(agg_daily)
        fold_objects = select_fold_objects(task_obj, mode=mode, max_folds=stage.max_folds)
        for fold_obj in fold_objects:
            step_count += 1
            train_samples_raw = recent_subset(fold_obj["train_samples"], stage.train_frac, min_keep=8)
            valid_samples_raw = recent_subset(fold_obj["valid_samples"], stage.valid_frac, min_keep=4)
            if mode == "holdout" and use_test_for_holdout:
                eval_raw = recent_subset(fold_obj["test_samples"], stage.eval_frac, min_keep=4)
            else:
                eval_raw = recent_subset(fold_obj["valid_samples"], stage.eval_frac, min_keep=4)

            epoch_offset = (step_count - 1) * max(1, stage.epochs)

            def epoch_logger(metrics: Dict[str, float], epoch: int) -> None:
                if trial is not None:
                    trial.report(float(metrics["valid_loss"]), step=epoch_offset + epoch)
                    if trial.should_prune():
                        optuna = maybe_import_optuna()
                        raise optuna.TrialPruned(f"Pruned at stage={stage.name}, task={task_name}, fold={fold_obj['fold_id']}, epoch={epoch}")
                if wandb_run is not None and wandb_run.active:
                    wandb_run.log({
                        f"{stage.name}/task": task_name,
                        f"{stage.name}/fold": int(fold_obj["fold_id"]),
                        f"{stage.name}/valid_loss": float(metrics["valid_loss"]),
                        f"{stage.name}/lr": float(metrics["lr"]),
                    }, step=epoch_offset + epoch)

            out = run_one_setting(
                model_name=str(params["model_name"]),
                train_samples_raw=train_samples_raw,
                valid_samples_raw=valid_samples_raw,
                test_like_samples_raw=eval_raw,
                agg_history_store=hist_store,
                task=task_name,
                device=device,
                fit_cfg=fit_cfg,
                reg_cfg=reg_cfg,
                quantiles=quantiles,
                hidden_dim=int(params["hidden_dim"]),
                dropout=float(params["dropout"]),
                num_heads=int(params.get("num_heads", 4)),
                num_sab_layers=int(params.get("num_sab_layers", 2)),
                num_seeds=int(params.get("num_seeds", 1)),
                num_inducing_points=int(params.get("num_inducing_points", 32)),
                recent_window=recent_window,
                lag_windows=lag_windows,
                stat_windows=stat_windows,
                monotone_quantiles=monotone_quantiles,
                cnn_num_blocks=int(params.get("cnn_num_blocks", 3)),
                cnn_kernel_size=int(params.get("cnn_kernel_size", 3)),
                epoch_logger=epoch_logger,
            )
            row = {
                "stage": stage.name,
                "task": task_name,
                "fold": int(fold_obj["fold_id"]),
                "model": str(params["model_name"]),
                "train_samples": len(train_samples_raw),
                "valid_samples": len(valid_samples_raw),
                "eval_samples": len(eval_raw),
                **{k: v for k, v in out.items() if k not in {"history"}},
            }
            all_rows.append(row)
            if trial is not None:
                running = objective_rows_mean(all_rows, quantiles)
                trial.set_user_attr("latest_running_objective", running)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return objective_rows_mean(all_rows, quantiles), all_rows


def safe_objective_factory(
    *,
    experiment: Dict[str, Any],
    stage1: StageBudget,
    mode: str,
    quantiles: Optional[Sequence[float]],
    device: torch.device,
    base_fit_cfg: FitConfig,
    reg_cfg: TrainRegularizationConfig,
    fixed_model_name: str,
    recent_window: int,
    lag_windows: Sequence[int],
    stat_windows: Sequence[int],
    monotone_quantiles: bool,
    wandb_run: WandbRunWrapper,
):
    optuna = maybe_import_optuna()

    def objective(trial: Any) -> float:
        params = suggest_params(trial, fixed_model_name)
        trial.set_user_attr("params", params)
        try:
            value, rows = evaluate_config_on_stage(
                experiment=experiment,
                stage=stage1,
                mode=mode,
                quantiles=quantiles,
                device=device,
                base_fit_cfg=base_fit_cfg,
                reg_cfg=reg_cfg,
                params=params,
                recent_window=recent_window,
                lag_windows=lag_windows,
                stat_windows=stat_windows,
                monotone_quantiles=monotone_quantiles,
                use_test_for_holdout=False,
                trial=trial,
                wandb_run=wandb_run,
            )
            trial.set_user_attr("rows", rows)
            if wandb_run.active:
                wandb_run.log({
                    "optuna/trial": int(trial.number),
                    "optuna/objective": float(value),
                    **{f"optuna/{k}": v for k, v in params.items()},
                }, step=int(trial.number))
            return float(value)
        except optuna.TrialPruned:
            raise
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cuda error" in message:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"OOM-pruned trial {trial.number}: {exc}") from exc
            raise

    return objective


def best_completed_trials(study: Any, top_k: int) -> List[Any]:
    trials = [t for t in study.trials if str(t.state).endswith("COMPLETE") and t.value is not None]
    trials.sort(key=lambda t: float(t.value))
    return trials[:top_k]


def trial_params_payload(trial: Any, default_model_name: str) -> Dict[str, Any]:
    params = dict(trial.user_attrs.get("params", {}))
    if "model_name" not in params:
        params["model_name"] = canonical_model_name(default_model_name or trial.params.get("model_name", "M3_FullSkuTemporalCNN"))
    params["stage1_objective"] = float(trial.value)
    params["trial_number"] = int(trial.number)
    return params


def stage2_confirm_topk(
    *,
    study: Any,
    top_k: int,
    experiment: Dict[str, Any],
    stage2: StageBudget,
    mode: str,
    quantiles: Optional[Sequence[float]],
    device: torch.device,
    base_fit_cfg: FitConfig,
    reg_cfg: TrainRegularizationConfig,
    fixed_model_name: str,
    recent_window: int,
    lag_windows: Sequence[int],
    stat_windows: Sequence[int],
    monotone_quantiles: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rank, trial in enumerate(best_completed_trials(study, top_k), start=1):
        params = trial_params_payload(trial, fixed_model_name)
        objective, stage_rows = evaluate_config_on_stage(
            experiment=experiment,
            stage=stage2,
            mode=mode,
            quantiles=quantiles,
            device=device,
            base_fit_cfg=base_fit_cfg,
            reg_cfg=reg_cfg,
            params=params,
            recent_window=recent_window,
            lag_windows=lag_windows,
            stat_windows=stat_windows,
            monotone_quantiles=monotone_quantiles,
            use_test_for_holdout=False,
        )
        summary = {
            "rank_from_stage1": rank,
            "trial_number": int(trial.number),
            "model_name": params["model_name"],
            "stage1_objective": float(trial.value),
            "stage2_objective": float(objective),
            "hidden_dim": int(params["hidden_dim"]),
            "dropout": float(params["dropout"]),
            "lr": float(params["lr"]),
            "weight_decay": float(params["weight_decay"]),
            "batch_size": int(params["batch_size"]),
            "cnn_num_blocks": int(params.get("cnn_num_blocks", 3)),
            "cnn_kernel_size": int(params.get("cnn_kernel_size", 3)),
            "tasks": ",".join(stage2.tasks),
        }
        task_df = pd.DataFrame(stage_rows)
        metric_key = "wspl" if quantiles is not None else "wrmsse"
        if not task_df.empty and metric_key in task_df.columns:
            task_means = task_df.groupby("task", as_index=False)[metric_key].mean()
            for _, r in task_means.iterrows():
                summary[f"task_{r['task']}_{metric_key}"] = float(r[metric_key])
        rows.append(summary)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["stage2_objective", "stage1_objective", "rank_from_stage1"], ignore_index=True)
    return df


def final_retrain_best(
    *,
    params: Dict[str, Any],
    experiment: Dict[str, Any],
    tasks: Sequence[str],
    mode: str,
    quantiles: Optional[Sequence[float]],
    device: torch.device,
    base_fit_cfg: FitConfig,
    reg_cfg: TrainRegularizationConfig,
    recent_window: int,
    lag_windows: Sequence[int],
    stat_windows: Sequence[int],
    monotone_quantiles: bool,
    final_epochs: int,
    final_patience: int,
) -> pd.DataFrame:
    final_stage = StageBudget(
        name="final",
        tasks=tuple(tasks),
        train_frac=1.0,
        valid_frac=1.0,
        eval_frac=1.0,
        epochs=final_epochs,
        patience=final_patience,
        max_folds=1,
    )
    final_fit = FitConfig(**{**asdict(base_fit_cfg), "epochs": final_epochs, "patience": final_patience})
    _, rows = evaluate_config_on_stage(
        experiment=experiment,
        stage=final_stage,
        mode=mode,
        quantiles=quantiles,
        device=device,
        base_fit_cfg=final_fit,
        reg_cfg=reg_cfg,
        params=params,
        recent_window=recent_window,
        lag_windows=lag_windows,
        stat_windows=stat_windows,
        monotone_quantiles=monotone_quantiles,
        use_test_for_holdout=(mode == "holdout"),
    )
    df = pd.DataFrame(rows)
    metric_key = "wspl" if quantiles is not None else "wrmsse"
    if not df.empty and metric_key in df.columns:
        summary_row = {
            "stage": "final_summary",
            "task": "ALL",
            "fold": 0,
            "model": params["model_name"],
            metric_key: float(df[metric_key].mean()),
        }
        if "wape" in df.columns:
            summary_row["wape"] = float(df["wape"].mean())
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Cheap multi-stage Optuna tuning for M0/M1/M3 on large M5 tasks.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./m5_optuna_outputs")
    parser.add_argument("--mode", type=str, default="holdout", choices=["rolling", "holdout"])
    parser.add_argument("--tasks", type=str, default="store_dept,store_cat,state_dept")
    parser.add_argument("--stage1-task", type=str, default="", help="Representative task for cheap screening. Defaults to the first task in --tasks.")
    parser.add_argument("--stage2-tasks", type=str, default="", help="Comma-separated confirmation tasks. Defaults to --tasks.")
    parser.add_argument("--model-name", type=str, default="M3_FullSkuTemporalCNN", help="Tune one model family at a time. Leave empty to let Optuna choose among M0/M1/M3.")
    parser.add_argument("--quantiles", type=str, default="", help="Comma-separated quantiles. Leave empty for point tuning.")
    parser.add_argument("--study-name", type=str, default="m5_multistage_tuning")
    parser.add_argument("--storage", type=str, default="", help="Optional Optuna storage URL, e.g. sqlite:///optuna.db")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"])
    parser.add_argument("--pruner", type=str, default="median", choices=["median", "hyperband", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-n-trials", type=int, default=24)
    parser.add_argument("--stage1-timeout-sec", type=int, default=0)
    parser.add_argument("--stage1-train-frac", type=float, default=0.25)
    parser.add_argument("--stage1-valid-frac", type=float, default=1.0)
    parser.add_argument("--stage1-eval-frac", type=float, default=1.0)
    parser.add_argument("--stage1-epochs", type=int, default=8)
    parser.add_argument("--stage1-patience", type=int, default=3)
    parser.add_argument("--stage1-max-folds", type=int, default=1)
    parser.add_argument("--stage2-top-k", type=int, default=6)
    parser.add_argument("--stage2-train-frac", type=float, default=0.6)
    parser.add_argument("--stage2-valid-frac", type=float, default=1.0)
    parser.add_argument("--stage2-eval-frac", type=float, default=1.0)
    parser.add_argument("--stage2-epochs", type=int, default=16)
    parser.add_argument("--stage2-patience", type=int, default=5)
    parser.add_argument("--stage2-max-folds", type=int, default=1)
    parser.add_argument("--final-epochs", type=int, default=24)
    parser.add_argument("--final-patience", type=int, default=6)
    parser.add_argument("--skip-final-stage", action="store_true")
    parser.add_argument("--t-hist", type=int, default=56)
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
    parser.add_argument("--lag-windows", type=str, default="7,14,28,56")
    parser.add_argument("--stat-windows", type=str, default="7,28,56")
    parser.add_argument("--disable-monotone-quantiles", action="store_true")
    parser.add_argument("--base-batch-size", type=int, default=16)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--base-weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-num-workers", type=int, default=0)
    parser.add_argument("--base-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU intra/inter-op thread limit for stable CPU training.")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="m5-cross-level-forecasting")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="optuna")
    parser.add_argument("--wandb-tags", type=str, default="m5,optuna,multistage")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    optuna = maybe_import_optuna()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch_runtime(device, cpu_threads=args.cpu_threads)

    tasks = parse_tasks(args.tasks)
    stage1_tasks = (args.stage1_task.strip(),) if args.stage1_task.strip() else (tasks[0],)
    stage2_tasks = parse_tasks(args.stage2_tasks) if args.stage2_tasks else tasks
    quantiles = [float(x) for x in args.quantiles.split(",")] if args.quantiles else None
    lag_windows = tuple(int(x) for x in args.lag_windows.split(",") if x.strip())
    stat_windows = tuple(int(x) for x in args.stat_windows.split(",") if x.strip())
    monotone_quantiles = not args.disable_monotone_quantiles

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

    base_fit_cfg = FitConfig(
        epochs=args.stage2_epochs,
        lr=args.base_lr,
        weight_decay=args.base_weight_decay,
        batch_size=args.base_batch_size,
        num_workers=args.base_num_workers,
        max_grad_norm=args.base_max_grad_norm,
        patience=args.stage2_patience,
        print_every=999999,
    )
    reg_cfg = TrainRegularizationConfig(
        use_permutation_consistency=False,
        permutation_prob=0.0,
        consistency_weight=0.0,
    )

    stage1 = StageBudget(
        name="stage1",
        tasks=stage1_tasks,
        train_frac=args.stage1_train_frac,
        valid_frac=args.stage1_valid_frac,
        eval_frac=args.stage1_eval_frac,
        epochs=args.stage1_epochs,
        patience=args.stage1_patience,
        max_folds=args.stage1_max_folds,
    )
    stage2 = StageBudget(
        name="stage2",
        tasks=stage2_tasks,
        train_frac=args.stage2_train_frac,
        valid_frac=args.stage2_valid_frac,
        eval_frac=args.stage2_eval_frac,
        epochs=args.stage2_epochs,
        patience=args.stage2_patience,
        max_folds=args.stage2_max_folds,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    run_config = {
        "cfg": asdict(cfg),
        "base_fit_cfg": asdict(base_fit_cfg),
        "mode": args.mode,
        "tasks": list(tasks),
        "stage1": asdict(stage1),
        "stage2": asdict(stage2),
        "final_epochs": args.final_epochs,
        "final_patience": args.final_patience,
        "seed": args.seed,
        "quantiles": quantiles,
        "cache_dir": cache_dir,
        "recent_window": args.recent_window,
        "lag_windows": list(lag_windows),
        "stat_windows": list(stat_windows),
        "monotone_quantiles": monotone_quantiles,
        "model_name": args.model_name,
        "skip_final_stage": bool(args.skip_final_stage),
    }
    save_json(run_config, os.path.join(args.output_dir, "tuning_run_config.json"))

    wandb_run = WandbRunWrapper(
        enabled=(args.use_wandb and args.wandb_mode != "disabled"),
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=parse_tags(args.wandb_tags),
        config=run_config,
        mode=args.wandb_mode,
        job_type="optuna_multistage",
    )

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

    sampler = optuna.samplers.TPESampler(seed=args.seed) if args.sampler == "tpe" else optuna.samplers.RandomSampler(seed=args.seed)
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=max(2, args.stage1_epochs // 2))
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(min_resource=2, max_resource=max(2, args.stage1_epochs), reduction_factor=3)
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=(args.storage or None),
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = safe_objective_factory(
        experiment=experiment,
        stage1=stage1,
        mode=args.mode,
        quantiles=quantiles,
        device=device,
        base_fit_cfg=base_fit_cfg,
        reg_cfg=reg_cfg,
        fixed_model_name=args.model_name,
        recent_window=args.recent_window,
        lag_windows=lag_windows,
        stat_windows=stat_windows,
        monotone_quantiles=monotone_quantiles,
        wandb_run=wandb_run,
    )
    study.optimize(objective, n_trials=args.stage1_n_trials, timeout=(args.stage1_timeout_sec or None), gc_after_trial=True)

    stage1_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    stage1_csv = os.path.join(args.output_dir, "stage1_trials.csv")
    stage1_df.to_csv(stage1_csv, index=False)

    stage2_df = stage2_confirm_topk(
        study=study,
        top_k=args.stage2_top_k,
        experiment=experiment,
        stage2=stage2,
        mode=args.mode,
        quantiles=quantiles,
        device=device,
        base_fit_cfg=base_fit_cfg,
        reg_cfg=reg_cfg,
        fixed_model_name=args.model_name,
        recent_window=args.recent_window,
        lag_windows=lag_windows,
        stat_windows=stat_windows,
        monotone_quantiles=monotone_quantiles,
    )
    stage2_csv = os.path.join(args.output_dir, "stage2_confirmation.csv")
    stage2_df.to_csv(stage2_csv, index=False)

    if not stage2_df.empty:
        best_row = stage2_df.iloc[0].to_dict()
        best_params = {
            "model_name": best_row["model_name"],
            "hidden_dim": int(best_row["hidden_dim"]),
            "dropout": float(best_row["dropout"]),
            "lr": float(best_row["lr"]),
            "weight_decay": float(best_row["weight_decay"]),
            "batch_size": int(best_row["batch_size"]),
            "cnn_num_blocks": int(best_row.get("cnn_num_blocks", 3)),
            "cnn_kernel_size": int(best_row.get("cnn_kernel_size", 3)),
            "recent_window": int(args.recent_window),
            "lag_windows": list(lag_windows),
            "stat_windows": list(stat_windows),
            "epochs": int(args.final_epochs),
            "patience": int(args.final_patience),
            "use_permreg": False,
            "stage1_objective": float(best_row["stage1_objective"]),
            "stage2_objective": float(best_row["stage2_objective"]),
            "trial_number": int(best_row["trial_number"]),
        }
    elif len(study.trials) > 0 and study.best_trial is not None:
        best_params = trial_params_payload(study.best_trial, args.model_name)
        best_params.update({
            "recent_window": int(args.recent_window),
            "lag_windows": list(lag_windows),
            "stat_windows": list(stat_windows),
            "epochs": int(args.final_epochs),
            "patience": int(args.final_patience),
            "use_permreg": False,
        })
    else:
        raise RuntimeError("No trials were completed; no best parameters available.")

    best_json = os.path.join(args.output_dir, "best_params.json")
    save_json(best_params, best_json)

    summary: Dict[str, Any] = {
        "stage1_trials_csv": stage1_csv,
        "stage2_confirmation_csv": stage2_csv,
        "best_params_json": best_json,
        "best_params": best_params,
    }

    if not args.skip_final_stage:
        final_df = final_retrain_best(
            params=best_params,
            experiment=experiment,
            tasks=tasks,
            mode=args.mode,
            quantiles=quantiles,
            device=device,
            base_fit_cfg=base_fit_cfg,
            reg_cfg=reg_cfg,
            recent_window=args.recent_window,
            lag_windows=lag_windows,
            stat_windows=stat_windows,
            monotone_quantiles=monotone_quantiles,
            final_epochs=args.final_epochs,
            final_patience=args.final_patience,
        )
        final_csv = os.path.join(args.output_dir, "final_retrain_results.csv")
        final_df.to_csv(final_csv, index=False)
        summary["final_retrain_results_csv"] = final_csv
        if not final_df.empty:
            metric_key = "wspl" if quantiles is not None else "wrmsse"
            summary["final_mean_metric"] = float(final_df.loc[final_df["task"] == "ALL", metric_key].iloc[0]) if (final_df["task"] == "ALL").any() else float(final_df[metric_key].mean())

    summary_json = os.path.join(args.output_dir, "multistage_tuning_summary.json")
    save_json(summary, summary_json)

    if wandb_run.active:
        wandb_run.summary_update(summary)
        wandb_run.log_dataframe("stage1_trials", stage1_df)
        wandb_run.log_dataframe("stage2_confirmation", stage2_df)
        wandb_run.log_artifact_path(args.output_dir, artifact_name=f"m5_multistage_tuning_{args.study_name}", artifact_type="results")
        wandb_run.finish()

    print("Multistage tuning summary:")
    print(summary)


if __name__ == "__main__":
    main()
