"""
Generate CD diagrams for 100% SKU sample with paper-consistent model names.

Reads 100pct series-level data, maps internal model names to paper names,
runs Friedman + Nemenyi tests, and produces CD diagram PNGs for both
WRMSSE (point forecasting) and WSPL (quantile forecasting).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad
from scipy.stats import norm

# ── Model name mapping: internal code → paper name ──────────────────────────
MODEL_NAME_MAP = {
    # Proposed (point)
    "SetTransformer": "Set Transformer",
    "DeepSets": "DeepSets",
    "M3_FullSkuTemporalCNN": "Gated Pooling",
    # Proposed (quantile)
    "SetTransformer_Q": "Set Transformer",
    "DeepSets_Q": "DeepSets",
    "M3_FullSkuTemporalCNN_Q": "Gated Pooling",
    # Ablation
    "M0_AggHistOnly": "AggHistOnly",
    "M1_AggHistFutureSummary": "AggHist Child-summary",
    # Benchmarks
    "BottomUpGlobalHistGB": "Bottom-up Global HistGB",
    "AggregateHistGB": "Aggregate HistGB",
    "AggregateElasticNet": "Aggregate Elastic Net",
    "ChildSummaryHistGB": "Child-summary HistGB",
    "ChildSummaryElasticNet": "Child-summary Elastic Net",
    "SeasonalNaive": "Seasonal Naive",
    "Reconcilation": "Reconcilation",
}

# Ablation models to keep (the rest are duplicates covered by benchmark/proposed)
ABLATION_KEEP = {"M0_AggHistOnly", "M1_AggHistFutureSummary"}

SERIES_METRIC_MAP = {
    "wrmsse": "rmsse",
    "wspl": "mean_spl",
}

# ── Data paths ────────────────────────────────────────────────────────────────
WORK = Path(__file__).resolve().parent.parent / "results" / "work"
FIGURES_DIR = WORK / "figures"

EXPERIMENTS = {
    "holdout": {
        "label": "Holdout",
        "benchmark_series": WORK / "100pct" / "03_benchmark_baselines" / "all_holdout_series_rows.csv",
        "proposed_series": WORK / "100pct" / "04_benchmark_proposed" / "proposed_all_holdout_series_rows.csv",
        "ablation_series": WORK / "100pct" / "02_ablation" / "all_ablation_series_rows_holdout.csv",
        "output_dir": WORK / "100pct" / "05_stat_tests",
    },
    "rolling": {
        "label": "Rolling",
        "benchmark_series": WORK / "100pct_rolling" / "03_benchmark_baselines" / "all_rolling_series_rows.csv",
        "proposed_series": WORK / "100pct_rolling" / "04_benchmark_proposed" / "proposed_all_rolling_series_rows.csv",
        "ablation_series": WORK / "100pct_rolling" / "02_ablation" / "all_ablation_series_rows_rolling.csv",
        "output_dir": WORK / "100pct_rolling" / "05_stat_tests",
    },
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> None:
    os.makedirs(path, exist_ok=True)


def infer_block_id(df: pd.DataFrame, block_cols: Sequence[str]) -> pd.Series:
    return df[list(block_cols)].astype(str).agg("__".join, axis=1)


def build_complete_series_matrix(
    series_rows_df: pd.DataFrame,
    metric: str,
    block_cols: Sequence[str],
    model_col: str = "model",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_name = SERIES_METRIC_MAP[metric]
    df = series_rows_df.loc[series_rows_df["metric"] == metric_name].copy()
    if df.empty:
        raise ValueError(f"No series rows found for metric '{metric_name}'.")
    df["block_id"] = infer_block_id(df, block_cols)
    agg = df.groupby(["block_id", model_col], as_index=False).agg(
        metric_value=("metric_value", "mean")
    )
    pivot = agg.pivot(index="block_id", columns=model_col, values="metric_value")
    complete = pivot.dropna(axis=0, how="any").copy()
    if complete.empty:
        raise ValueError("No complete series blocks remain after aligning models.")
    if complete.shape[1] < 3:
        raise ValueError("Friedman/Nemenyi requires at least 3 models.")
    ranks = complete.rank(axis=1, method="average", ascending=True)
    return complete, ranks


def friedman_test(block_matrix: pd.DataFrame) -> Tuple[float, float]:
    samples = [block_matrix[c].to_numpy(dtype=float) for c in block_matrix.columns]
    stat, p = stats.friedmanchisquare(*samples)
    return float(stat), float(p)


# ── Studentized range distribution for infinite df (numerical integration) ──

def _studentized_range_sf(q: float, k: int) -> float:
    """Survival function of studentized range with k groups, infinite df."""
    if q <= 0:
        return 1.0

    def integrand(x: float) -> float:
        phi_x = norm.pdf(x)
        Phi_x = norm.cdf(x)
        Phi_xmq = norm.cdf(x - q)
        diff = Phi_x - Phi_xmq
        if diff <= 0:
            return 0.0
        return float(k * phi_x * (diff ** (k - 1)))

    # Integrate over a reasonable range; the integrand decays quickly
    result, _ = quad(integrand, -np.inf, np.inf, limit=200)
    return float(1.0 - max(0.0, min(1.0, result)))


def _studentized_range_isf(alpha: float, k: int) -> float:
    """Inverse survival function: find q such that SF(q) = alpha."""
    from scipy.optimize import brentq
    # Bracket: q is between 0 and about 6 for practical alpha values
    return float(brentq(lambda q: _studentized_range_sf(q, k) - alpha, 0.0, 10.0))


def nemenyi_pairwise_pvalues(
    mean_ranks: pd.Series, n_blocks: int, alpha: float = 0.10
) -> Tuple[pd.DataFrame, float]:
    models_list = list(mean_ranks.index)
    k = len(models_list)
    se = math.sqrt(k * (k + 1) / (6.0 * n_blocks))
    q_alpha = float(_studentized_range_isf(alpha, k) / math.sqrt(2.0))
    cd = q_alpha * se
    pvals = pd.DataFrame(np.ones((k, k), dtype=float), index=models_list, columns=models_list)
    for i, mi in enumerate(models_list):
        for j, mj in enumerate(models_list):
            if i == j:
                continue
            diff = abs(float(mean_ranks.loc[mi]) - float(mean_ranks.loc[mj]))
            q_stat = diff * math.sqrt(2.0) / se
            pvals.loc[mi, mj] = float(_studentized_range_sf(q_stat, k))
    return pvals, cd


def plot_cd_diagram(
    mean_ranks: pd.Series,
    cd: float,
    metric_label: str,
    n_blocks: int,
    output_path: str,
    title_prefix: str = "",
) -> None:
    ordered = mean_ranks.sort_values(ascending=True)
    models_list = ordered.index.tolist()
    x = ordered.to_numpy(dtype=float)
    y = np.arange(len(models_list), 0, -1, dtype=float)
    k = len(models_list)

    fig_h = max(3.0, 0.55 * k + 1.6)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    for i in range(k):
        ax.plot(
            [x[i] - cd / 2, x[i] + cd / 2],
            [y[i], y[i]],
            color="k",
            linewidth=2.5,
            alpha=0.4,
        )

    ax.scatter(x, y, s=45, zorder=5)
    for xi, yi, label in zip(x, y, models_list):
        ax.text(xi, yi + 0.1, label, ha="center", va="bottom", fontsize=10)

    xmin = min(1.0, float(np.min(x)) - 0.25)
    xmax = max(float(k), float(np.max(x)) + cd / 2 + 0.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.5, k + 1.3)
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_yticks([])

    prefix = f"{title_prefix} — " if title_prefix else ""
    title = f"{prefix}Critical-difference rank plot for {metric_label.upper()} (N={n_blocks} blocks)"
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Legend for CD bar
    y_cd = k + 0.7
    x0 = xmin + 0.15
    x1 = x0 + cd
    ax.plot([x0, x1], [y_cd, y_cd], color="k", linewidth=2.5, alpha=0.4)
    ax.plot([x0, x0], [y_cd - 0.10, y_cd + 0.10], color="k", linewidth=1.0)
    ax.plot([x1, x1], [y_cd - 0.10, y_cd + 0.10], color="k", linewidth=1.0)
    ax.text(
        (x0 + x1) / 2.0,
        y_cd + 0.12,
        f"CD = {cd:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CD diagram: {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def run_one_metric(
    metric: str,  # "wrmsse" or "wspl"
    bench_df: pd.DataFrame,
    prop_df: pd.DataFrame,
    abl_df: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
    tag: str = "",  # e.g. "holdout" or "rolling"
    alpha: float = 0.10,
) -> Tuple[pd.Series, float, int]:
    """Run Friedman+Nemenyi for one metric and save CD diagram PNG."""
    metric_label = "wrmsse" if metric == "wrmsse" else "mean_spl"
    display_label = "WRMSSE" if metric == "wrmsse" else "WSPL"
    title_prefix = "Holdout" if tag == "holdout" else "Rolling-origin" if tag == "rolling" else ""

    # Filter & rename
    abl_filt = abl_df[abl_df["model"].isin(ABLATION_KEEP)].copy()
    combined = pd.concat([bench_df, prop_df, abl_filt], ignore_index=True)
    combined["model"] = combined["model"].map(MODEL_NAME_MAP)

    block_cols = ["task", "agg_id"]
    block_matrix, rank_matrix = build_complete_series_matrix(combined, metric, block_cols)

    n_blocks = block_matrix.shape[0]
    n_models = block_matrix.shape[1]
    print(f"\n{'='*60}")
    print(f"Experiment: {tag}, Metric: {display_label}")
    print(f"  Blocks: {n_blocks}, Models: {n_models}")
    print(f"  Models: {list(block_matrix.columns)}")

    # Friedman test
    chi2, p_val = friedman_test(block_matrix)
    print(f"  Friedman chi2={chi2:.4f}, p={p_val:.4g}")

    # Nemenyi
    mean_ranks = rank_matrix.mean(axis=0)
    pvals, cd = nemenyi_pairwise_pvalues(mean_ranks, n_blocks, alpha=alpha)
    print(f"  CD (alpha={alpha}): {cd:.4f}")
    print(f"  Mean ranks:\n{mean_ranks.sort_values().to_string()}")

    # Save summary CSV
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

    # Save CSVs
    ensure_dir(output_dir)
    tag_slug = f"_{tag}" if tag else ""
    summary_df.to_csv(output_dir / f"Table_stats_{metric_label}_series{tag_slug}.csv", index=False)
    rank_matrix.to_csv(output_dir / f"rank_matrix_{metric_label}_series{tag_slug}.csv")
    block_matrix.to_csv(output_dir / f"block_matrix_{metric_label}_series{tag_slug}.csv")
    pvals.to_csv(output_dir / f"nemenyi_pairwise_pvalues_{metric_label}_series{tag_slug}.csv")

    # Generate CD diagram PNG
    ensure_dir(figures_dir)
    png_name = f"Figure_cd_diagram_{metric_label}_series{tag_slug}.png"
    png_path = figures_dir / png_name

    plot_cd_diagram(mean_ranks, cd, display_label, n_blocks, str(png_path),
                    title_prefix=title_prefix)

    # Also save a copy to the stat_tests output dir for consistency
    plot_cd_diagram(mean_ranks, cd, display_label, n_blocks,
                    str(output_dir / f"Figure_cd_diagram_{metric_label}_series.png"),
                    title_prefix=title_prefix)

    return mean_ranks, cd, n_blocks


def main() -> None:
    for tag, cfg in EXPERIMENTS.items():
        print(f"\n{'#'*60}")
        print(f"Processing: {cfg['label']} ({tag})")
        print(f"{'#'*60}")

        bench_df = pd.read_csv(cfg["benchmark_series"])
        prop_df = pd.read_csv(cfg["proposed_series"])
        abl_df = pd.read_csv(cfg["ablation_series"])

        for fname, df in [("benchmark", bench_df), ("proposed", prop_df), ("ablation", abl_df)]:
            print(f"  {fname}: {len(df)} rows, models={sorted(df['model'].unique())}")

        # Run both metrics
        for metric in ["wrmsse", "wspl"]:
            run_one_metric(
                metric=metric,
                bench_df=bench_df,
                prop_df=prop_df,
                abl_df=abl_df,
                output_dir=cfg["output_dir"],
                figures_dir=FIGURES_DIR,
                tag=tag,
                alpha=0.10,
            )

    print(f"\nDone. Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
