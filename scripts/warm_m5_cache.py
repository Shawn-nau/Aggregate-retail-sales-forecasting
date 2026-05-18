from __future__ import annotations

import argparse

from prepare_m5_experiments import (
    M5ExperimentConfig,
    build_m5_experiment_samples,
    default_cache_dir,
    print_experiment_summary,
)


def parse_tasks(s: str):
    return tuple(x.strip() for x in s.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute and cache M5 preprocessing artifacts for later experiments.")
    parser.add_argument("--data-dir", type=str, required=True)
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
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="Build holdout/final-train/internal-valid/test sample caches.")
    parser.add_argument("--rolling", action="store_true", help="Build rolling-fold sample caches.")
    parser.add_argument("--outer", action="store_true", help="Build outer train/valid sample caches.")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()

    cfg = M5ExperimentConfig(
        data_dir=args.data_dir,
        t_hist=args.t_hist,
        horizon=args.horizon,
        tasks=parse_tasks(args.tasks),
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

    if not any([args.outer, args.holdout, args.rolling]):
        args.holdout = True
        args.rolling = True

    experiment = build_m5_experiment_samples(
        raw_data=None,
        cfg=cfg,
        need_outer=args.outer,
        need_holdout=args.holdout,
        need_rolling=args.rolling,
        cache_dir=cache_dir,
        force_rebuild=args.force_rebuild_cache,
        progress=(not args.quiet_progress),
    )
    print_experiment_summary(experiment)
    print(f"\nCache ready at: {cache_dir}")


if __name__ == "__main__":
    main()
