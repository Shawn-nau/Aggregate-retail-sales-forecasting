"""
Generate the paper's rolling-origin CD diagrams (Figures 4-5) at the
fold level: one aligned block per aggregate series per rolling fold
(121 aggregate series x 5 rolling folds = 605 blocks).

The series-level rolling diagrams produced by generate_cd_diagrams_100pct.py
(averaging per-origin metrics within task x agg_id blocks, 121 blocks) are
kept as a supplementary analysis; the PAPER's rolling figures use the
605-block fold-level analysis generated here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from generate_cd_diagrams_100pct import (
    ABLATION_KEEP,
    FIGURES_DIR,
    MODEL_NAME_MAP,
    WORK,
    build_complete_series_matrix,
    ensure_dir,
    friedman_test,
    nemenyi_pairwise_pvalues,
    plot_cd_diagram,
)

ROLLING = WORK / "100pct_rolling"
OUTPUT_DIR = ROLLING / "05_stat_tests"

BENCH_SERIES = ROLLING / "03_benchmark_baselines" / "all_rolling_series_rows.csv"
PROPOSED_SERIES = ROLLING / "04_benchmark_proposed" / "proposed_all_rolling_series_rows.csv"
ABLATION_SERIES = ROLLING / "02_ablation" / "all_ablation_series_rows_rolling.csv"

EXPECTED_N_BLOCKS = 605  # 121 aggregate series x 5 rolling folds

BLOCK_COLS = ["task", "agg_id", "fold"]
TITLE_PREFIX = "Rolling-origin (fold-level)"


def run_one_metric(metric: str, alpha: float = 0.10):
    """Run Friedman+Nemenyi over fold-level blocks for one metric."""
    metric_label = "wrmsse" if metric == "wrmsse" else "mean_spl"
    display_label = "WRMSSE" if metric == "wrmsse" else "WSPL"

    bench_df = pd.read_csv(BENCH_SERIES)
    prop_df = pd.read_csv(PROPOSED_SERIES)
    abl_df = pd.read_csv(ABLATION_SERIES)

    abl_filt = abl_df[abl_df["model"].isin(ABLATION_KEEP)].copy()
    combined = pd.concat([bench_df, prop_df, abl_filt], ignore_index=True)
    combined["model"] = combined["model"].map(MODEL_NAME_MAP)

    block_matrix, rank_matrix = build_complete_series_matrix(combined, metric, BLOCK_COLS)

    n_blocks = block_matrix.shape[0]
    n_models = block_matrix.shape[1]
    print(f"\n{'='*60}")
    print(f"Rolling fold-level, Metric: {display_label}")
    print(f"  Blocks: {n_blocks} (expected {EXPECTED_N_BLOCKS}), Models: {n_models}")
    print(f"  Models: {list(block_matrix.columns)}")
    if n_blocks != EXPECTED_N_BLOCKS:
        print(f"  WARNING: block count differs from the expected {EXPECTED_N_BLOCKS}")

    chi2, p_val = friedman_test(block_matrix)
    print(f"  Friedman chi2={chi2:.4f}, p={p_val:.4g}")

    mean_ranks = rank_matrix.mean(axis=0)
    pvals, cd = nemenyi_pairwise_pvalues(mean_ranks, n_blocks, alpha=alpha)
    print(f"  CD (alpha={alpha}): {cd:.4f}")
    print(f"  Mean ranks:\n{mean_ranks.sort_values().to_string()}")

    summary_rows = []
    for m in mean_ranks.sort_values().index:
        wins = 0
        losses = 0
        for other in mean_ranks.index:
            if other == m:
                continue
            if pvals.loc[m, other] < alpha:
                if mean_ranks.loc[m] < mean_ranks.loc[other]:
                    wins += 1
                else:
                    losses += 1
        summary_rows.append({
            "model": m,
            f"{metric_label}_mean": float(block_matrix[m].mean()),
            f"{metric_label}_std": float(block_matrix[m].std(ddof=1)),
            "avg_rank": float(mean_ranks.loc[m]),
            "nemenyi_sig_wins": wins,
            "nemenyi_sig_losses": losses,
        })
    summary_df = pd.DataFrame(summary_rows)

    ensure_dir(OUTPUT_DIR)
    tag = "rolling_foldlevel"
    summary_df.to_csv(OUTPUT_DIR / f"Table_stats_{metric_label}_series_{tag}.csv", index=False)
    rank_matrix.to_csv(OUTPUT_DIR / f"rank_matrix_{metric_label}_series_{tag}.csv")
    block_matrix.to_csv(OUTPUT_DIR / f"block_matrix_{metric_label}_series_{tag}.csv")
    pvals.to_csv(OUTPUT_DIR / f"nemenyi_pairwise_pvalues_{metric_label}_series_{tag}.csv")

    ensure_dir(FIGURES_DIR)
    png_path = FIGURES_DIR / f"Figure_cd_diagram_{metric_label}_series_{tag}.png"
    plot_cd_diagram(mean_ranks, cd, display_label, n_blocks, str(png_path),
                    title_prefix=TITLE_PREFIX)

    return mean_ranks, cd, n_blocks


def main() -> None:
    for metric in ["wrmsse", "wspl"]:
        run_one_metric(metric, alpha=0.10)
    print(f"\nDone. Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
