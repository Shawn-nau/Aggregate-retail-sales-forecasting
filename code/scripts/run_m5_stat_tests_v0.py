from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range


SERIES_METRIC_MAP = {
    "wrmsse": "rmsse",
    "wspl": "mean_spl",
    "rmsse": "rmsse",
    "mean_spl": "mean_spl",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def _fmt(x: float, ndigits: int = 4) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{ndigits}f}"


def load_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or path == "":
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find CSV: {path}")
    return pd.read_csv(p)


def merge_rows(*parts: Optional[pd.DataFrame]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for part in parts:
        if part is not None and not part.empty:
            dfs.append(part.copy())
    if not dfs:
        raise ValueError("At least one non-empty CSV must be provided.")
    return pd.concat(dfs, axis=0, ignore_index=True)


def split_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def infer_block_id(df: pd.DataFrame, block_cols: Sequence[str]) -> pd.Series:
    missing = [c for c in block_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing block columns in input CSV: {missing}")
    if len(block_cols) == 1:
        return df[block_cols[0]].astype(str)
    return df[list(block_cols)].astype(str).agg("__".join, axis=1)


def build_complete_block_matrix(rows_df: pd.DataFrame, metric: str, block_cols: Sequence[str], model_col: str = "model") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metric not in rows_df.columns:
        raise ValueError(f"Metric '{metric}' not found in rows CSV.")
    df = rows_df.loc[rows_df[metric].notna()].copy()
    if df.empty:
        raise ValueError(f"No non-null rows found for metric '{metric}'.")
    df["block_id"] = infer_block_id(df, block_cols)

    dup_counts = df.groupby(["block_id", model_col]).size()
    if (dup_counts > 1).any():
        df = df.groupby(["block_id", model_col], as_index=False).agg(**{metric: (metric, "mean")})
    else:
        df = df[["block_id", model_col, metric]].copy()

    pivot = df.pivot(index="block_id", columns=model_col, values=metric)
    complete = pivot.dropna(axis=0, how="any").copy()
    if complete.empty:
        raise ValueError(
            "No complete blocks remain after aligning models. "
            "Check whether every model was evaluated on the same instances."
        )
    if complete.shape[1] < 3:
        raise ValueError("Friedman/Nemenyi requires at least 3 models with complete coverage.")
    ranks = complete.rank(axis=1, method="average", ascending=True)
    return complete, ranks


def build_complete_series_matrix(series_rows_df: pd.DataFrame, metric: str, block_cols: Sequence[str], model_col: str = "model") -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_name = SERIES_METRIC_MAP.get(metric, metric)
    if metric_name in {"rmsse", "mean_spl"}:
        if "metric" not in series_rows_df.columns or "metric_value" not in series_rows_df.columns:
            raise ValueError("Series rows CSV must contain 'metric' and 'metric_value' columns.")
        df = series_rows_df.loc[series_rows_df["metric"] == metric_name].copy()
        metric_col = "metric_value"
    else:
        raise ValueError(
            f"Series-level tests do not support metric '{metric}'. "
            "Use wrmsse/rmsse for point forecasts or wspl/mean_spl for quantiles."
        )
    if df.empty:
        raise ValueError(f"No series rows found for series metric '{metric_name}'.")
    df["block_id"] = infer_block_id(df, block_cols)
    agg = (
        df.groupby(["block_id", model_col], as_index=False)
        .agg(metric_value=(metric_col, "mean"))
    )
    pivot = agg.pivot(index="block_id", columns=model_col, values="metric_value")
    complete = pivot.dropna(axis=0, how="any").copy()
    if complete.empty:
        raise ValueError(
            "No complete series blocks remain after aligning models. "
            "Check whether every model has the same task×series coverage."
        )
    if complete.shape[1] < 3:
        raise ValueError("Friedman/Nemenyi requires at least 3 models with complete coverage.")
    ranks = complete.rank(axis=1, method="average", ascending=True)
    return complete, ranks


def friedman_test(block_matrix: pd.DataFrame) -> Tuple[float, float]:
    samples = [block_matrix[c].to_numpy(dtype=float) for c in block_matrix.columns]
    stat, p = stats.friedmanchisquare(*samples)
    return float(stat), float(p)


def nemenyi_pairwise_pvalues(mean_ranks: pd.Series, n_blocks: int, alpha: float = 0.05) -> Tuple[pd.DataFrame, float]:
    models = list(mean_ranks.index)
    k = len(models)
    se = math.sqrt(k * (k + 1) / (6.0 * n_blocks))
    q_alpha = float(studentized_range.isf(alpha, k, np.inf) / math.sqrt(2.0))
    cd = q_alpha * se

    pvals = pd.DataFrame(np.ones((k, k), dtype=float), index=models, columns=models)
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i == j:
                pvals.loc[mi, mj] = 1.0
                continue
            diff = abs(float(mean_ranks.loc[mi]) - float(mean_ranks.loc[mj]))
            q_stat = diff * math.sqrt(2.0) / se
            pvals.loc[mi, mj] = float(studentized_range.sf(q_stat, k, np.inf))
    return pvals, cd


def build_summary_table(block_matrix: pd.DataFrame, ranks: pd.DataFrame, pairwise_p: pd.DataFrame, alpha: float, metric_label: str) -> pd.DataFrame:
    mean_metric = block_matrix.mean(axis=0)
    std_metric = block_matrix.std(axis=0, ddof=1)
    mean_rank = ranks.mean(axis=0)

    order = mean_rank.sort_values(ascending=True).index.tolist()
    rows = []
    for m in order:
        wins = 0
        losses = 0
        for other in order:
            if other == m:
                continue
            if pairwise_p.loc[m, other] < alpha:
                if mean_rank.loc[m] < mean_rank.loc[other]:
                    wins += 1
                elif mean_rank.loc[m] > mean_rank.loc[other]:
                    losses += 1
        rows.append({
            "model": m,
            f"{metric_label}_mean": float(mean_metric.loc[m]),
            f"{metric_label}_std": float(std_metric.loc[m]) if not pd.isna(std_metric.loc[m]) else np.nan,
            "avg_rank": float(mean_rank.loc[m]),
            "nemenyi_sig_wins": int(wins),
            "nemenyi_sig_losses": int(losses),
        })
    return pd.DataFrame(rows)


def summary_to_latex(
    summary_df: pd.DataFrame,
    metric_label: str,
    analysis_level: str,
    block_desc: str,
    n_blocks: int,
    friedman_stat: float,
    friedman_p: float,
    cd: float,
    alpha: float,
    caution_note: str = "",
) -> str:
    if summary_df.empty:
        return f"% Statistical test table for {metric_label}: no data available\n"

    lines: List[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Relative performance comparison by {metric_label.upper()} using the Friedman test with post-hoc Nemenyi analysis.}}")
    lines.append(rf"\label{{tab:stats_{metric_label}}}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(rf"Model & Mean {metric_label.upper()} & Std. & Avg. rank & Sig. wins & Sig. losses \\")
    lines.append(r"\midrule")
    for _, row in summary_df.iterrows():
        lines.append(
            f"{_latex_escape(row['model'])} & "
            f"{_fmt(row[f'{metric_label}_mean'], 4)} & "
            f"{_fmt(row[f'{metric_label}_std'], 4)} & "
            f"{_fmt(row['avg_rank'], 3)} & "
            f"{int(row['nemenyi_sig_wins'])} & "
            f"{int(row['nemenyi_sig_losses'])} "
            + r"\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    note = (
        rf"\footnotesize \item The Friedman test uses {n_blocks} aligned {analysis_level}-level blocks ({_latex_escape(block_desc)}). "
        rf"The test statistic is $\chi^2={friedman_stat:.4f}$ with $p={friedman_p:.4g}$. "
        rf"Average ranks are computed with lower {metric_label.upper()} indicating better performance. "
        rf"Nemenyi significant wins/losses are counted at $\alpha={alpha:.2f}$ using critical difference $CD={cd:.4f}$."
    )
    if caution_note:
        note += " " + caution_note
    lines.append(note)
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def plot_cd_diagram(mean_ranks: pd.Series, cd: float, metric_label: str, n_blocks: int, output_path: str) -> None:
    ordered = mean_ranks.sort_values(ascending=True)
    models = ordered.index.tolist()
    x = ordered.to_numpy(dtype=float)
    y = np.arange(len(models), 0, -1, dtype=float)
    k = len(models)

    fig_h = max(3.0, 0.55 * len(models) + 1.6)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.scatter(x, y, s=45)
    for xi, yi, label in zip(x, y, models):
        ax.text(xi + 0.03, yi, label, va="center", fontsize=10)

    xmin = min(1.0, float(np.min(x)) - 0.25)
    xmax = max(float(k), float(np.max(x)) + cd + 0.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.5, len(models) + 1.3)
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_yticks([])
    ax.set_title(f"Critical-difference style rank plot for {metric_label.upper()} (N={n_blocks} blocks)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    y_cd = len(models) + 0.7
    x0 = xmin + 0.15
    x1 = x0 + cd
    ax.plot([x0, x1], [y_cd, y_cd], linewidth=2.0)
    ax.plot([x0, x0], [y_cd - 0.10, y_cd + 0.10], linewidth=2.0)
    ax.plot([x1, x1], [y_cd - 0.10, y_cd + 0.10], linewidth=2.0)
    ax.text((x0 + x1) / 2.0, y_cd + 0.12, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    ensure_dir(str(Path(output_path).parent))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Friedman and post-hoc Nemenyi tests on M5 forecast results.")
    parser.add_argument("--benchmark-rows", type=str, default="", help="Aggregate/block-level CSV from run_m5_full_benchmark.py")
    parser.add_argument("--proposed-rows", type=str, default="", help="Aggregate/block-level CSV from run_m5_proposed_models.py")
    parser.add_argument("--benchmark-series-rows", type=str, default="", help="Series-level CSV from run_m5_full_benchmark.py")
    parser.add_argument("--proposed-series-rows", type=str, default="", help="Series-level CSV from run_m5_proposed_models.py")
    parser.add_argument("--metric", type=str, default="wspl", choices=["wrmsse", "wape", "wspl", "rmsse", "mean_spl"], help="Metric for the rank-based test. Default is WSPL; with series rows present this uses series-level mean pinball for aligned rank testing.")
    parser.add_argument("--analysis-level", type=str, default="auto", choices=["auto", "series", "block"], help="Use individual-series ranks when series rows are available.")
    parser.add_argument("--block-cols", type=str, default="task,fold", help="Columns defining aligned aggregate blocks for block-level tests.")
    parser.add_argument("--series-block-cols", type=str, default="task,agg_id", help="Columns defining individual series blocks for series-level tests.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    benchmark_rows = load_csv(args.benchmark_rows)
    proposed_rows = load_csv(args.proposed_rows)
    benchmark_series_rows = load_csv(args.benchmark_series_rows)
    proposed_series_rows = load_csv(args.proposed_series_rows)

    rows = merge_rows(benchmark_rows, proposed_rows) if (benchmark_rows is not None or proposed_rows is not None) else pd.DataFrame()
    series_rows = merge_rows(benchmark_series_rows, proposed_series_rows) if (benchmark_series_rows is not None or proposed_series_rows is not None) else pd.DataFrame()

    if not rows.empty:
        rows.to_csv(os.path.join(args.output_dir, f"combined_rows_{args.metric}.csv"), index=False)
    if not series_rows.empty:
        series_rows.to_csv(os.path.join(args.output_dir, f"combined_series_rows_{args.metric}.csv"), index=False)

    use_series = False
    if args.analysis_level == "series":
        use_series = True
    elif args.analysis_level == "auto":
        use_series = not series_rows.empty and args.metric in SERIES_METRIC_MAP

    if use_series:
        block_cols = split_csv(args.series_block_cols)
        block_matrix, ranks = build_complete_series_matrix(series_rows, metric=args.metric, block_cols=block_cols)
        analysis_level = "series"
        metric_label = SERIES_METRIC_MAP.get(args.metric, args.metric)
        block_desc = ",".join(block_cols)
    else:
        if rows.empty:
            raise ValueError("Block-level analysis requested but no aggregate rows CSV was provided.")
        block_cols = split_csv(args.block_cols)
        block_matrix, ranks = build_complete_block_matrix(rows, metric=args.metric, block_cols=block_cols)
        analysis_level = "block"
        metric_label = args.metric
        block_desc = ",".join(block_cols)

    block_matrix.to_csv(os.path.join(args.output_dir, f"block_matrix_{metric_label}_{analysis_level}.csv"))
    ranks.to_csv(os.path.join(args.output_dir, f"rank_matrix_{metric_label}_{analysis_level}.csv"))

    friedman_stat, friedman_p = friedman_test(block_matrix)
    mean_ranks = ranks.mean(axis=0).sort_values(ascending=True)
    pairwise_p, cd = nemenyi_pairwise_pvalues(mean_ranks, n_blocks=block_matrix.shape[0], alpha=args.alpha)
    pairwise_p.to_csv(os.path.join(args.output_dir, f"nemenyi_pairwise_pvalues_{metric_label}_{analysis_level}.csv"))

    summary_df = build_summary_table(block_matrix, ranks, pairwise_p, alpha=args.alpha, metric_label=metric_label)
    summary_df.to_csv(os.path.join(args.output_dir, f"Table_stats_{metric_label}_{analysis_level}.csv"), index=False)

    overview = pd.DataFrame([
        {
            "metric_requested": args.metric,
            "metric_used": metric_label,
            "analysis_level": analysis_level,
            "block_columns": block_desc,
            "n_blocks": int(block_matrix.shape[0]),
            "n_models": int(block_matrix.shape[1]),
            "friedman_chi2": friedman_stat,
            "friedman_p": friedman_p,
            "critical_difference": cd,
            "alpha": args.alpha,
        }
    ])
    overview.to_csv(os.path.join(args.output_dir, f"friedman_overview_{metric_label}_{analysis_level}.csv"), index=False)

    caution_note = ""
    if block_matrix.shape[0] < 10:
        caution_note = (
            "Because the number of aligned blocks is modest, the test may have limited power; "
            "interpret non-significant differences cautiously."
        )

    latex = summary_to_latex(
        summary_df=summary_df,
        metric_label=metric_label,
        analysis_level=analysis_level,
        block_desc=block_desc,
        n_blocks=int(block_matrix.shape[0]),
        friedman_stat=friedman_stat,
        friedman_p=friedman_p,
        cd=cd,
        alpha=args.alpha,
        caution_note=caution_note,
    )
    Path(os.path.join(args.output_dir, f"Table_stats_{metric_label}_{analysis_level}.tex")).write_text(latex, encoding="utf-8")

    plot_cd_diagram(
        mean_ranks=mean_ranks,
        cd=cd,
        metric_label=metric_label,
        n_blocks=int(block_matrix.shape[0]),
        output_path=os.path.join(args.output_dir, f"Figure_cd_diagram_{metric_label}_{analysis_level}.png"),
    )

    notes = [
        "Statistical test workflow summary",
        "",
        f"Requested metric: {args.metric}",
        f"Metric used: {metric_label}",
        f"Analysis level: {analysis_level}",
        f"Block columns: {block_cols}",
        f"Number of complete blocks: {block_matrix.shape[0]}",
        f"Number of compared models: {block_matrix.shape[1]}",
        f"Friedman chi-square: {friedman_stat:.6f}",
        f"Friedman p-value: {friedman_p:.6g}",
        f"Nemenyi critical difference (alpha={args.alpha:.2f}): {cd:.6f}",
    ]
    if use_series:
        notes += [
            "",
            "Series-level protocol:",
            "Each task×series block is formed by averaging the per-origin series metric across rolling folds/origins for each model, then ranking models within that block.",
        ]
    if caution_note:
        notes += ["", "Caution:", caution_note]
    Path(os.path.join(args.output_dir, f"README_stats_{metric_label}_{analysis_level}.md")).write_text("\n".join(notes) + "\n", encoding="utf-8")

    print("Done. Statistical test outputs written to:")
    print(args.output_dir)


if __name__ == "__main__":
    main()
