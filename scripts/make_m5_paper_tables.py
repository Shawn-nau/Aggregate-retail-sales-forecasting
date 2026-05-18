from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fmt(x: float, ndigits: int = 4) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{ndigits}f}"


def _latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def _bold_best_and_underline_second(values: pd.Series, lower_is_better: bool = True, ndigits: int = 4) -> List[str]:
    vals = values.astype(float)
    order = vals.sort_values(ascending=lower_is_better)
    best_idx = order.index[0] if len(order) >= 1 else None
    second_idx = order.index[1] if len(order) >= 2 else None

    out = []
    for idx, v in vals.items():
        s = _fmt(v, ndigits)
        if idx == best_idx:
            s = r"\textbf{" + s + "}"
        elif idx == second_idx:
            s = r"\underline{" + s + "}"
        out.append(s)
    return out


def load_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or path == "":
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find CSV: {path}")
    return pd.read_csv(p)


def merge_rows(benchmark_rows: Optional[pd.DataFrame], proposed_rows: Optional[pd.DataFrame]) -> pd.DataFrame:
    parts = []
    if benchmark_rows is not None:
        parts.append(benchmark_rows.copy())
    if proposed_rows is not None:
        parts.append(proposed_rows.copy())
    if not parts:
        raise ValueError("At least one of benchmark_rows or proposed_rows must be provided.")
    return pd.concat(parts, axis=0, ignore_index=True)


def summarize_point_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    if "wrmsse" not in rows_df.columns:
        return pd.DataFrame(columns=["task", "model", "wrmsse_mean", "wrmsse_std", "wape_mean", "wape_std", "n_folds"])
    point_df = rows_df.loc[rows_df["wrmsse"].notna()].copy()
    if point_df.empty:
        return pd.DataFrame(columns=["task", "model", "wrmsse_mean", "wrmsse_std", "wape_mean", "wape_std", "n_folds"])
    return (
        point_df.groupby(["task", "model"], as_index=False)
        .agg(
            wrmsse_mean=("wrmsse", "mean"),
            wrmsse_std=("wrmsse", "std"),
            wape_mean=("wape", "mean"),
            wape_std=("wape", "std"),
            n_folds=("fold", "count"),
        )
    )


def summarize_quantile_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    if "wspl" not in rows_df.columns:
        return pd.DataFrame(columns=["task", "model", "wspl_mean", "wspl_std", "n_folds"])
    q_df = rows_df.loc[rows_df["wspl"].notna()].copy()
    if q_df.empty:
        return pd.DataFrame(columns=["task", "model", "wspl_mean", "wspl_std", "n_folds"])
    return (
        q_df.groupby(["task", "model"], as_index=False)
        .agg(
            wspl_mean=("wspl", "mean"),
            wspl_std=("wspl", "std"),
            n_folds=("fold", "count"),
        )
    )


def build_main_point_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    table = summary_df.pivot(index="model", columns="task", values="wrmsse_mean").copy()
    if not table.empty:
        table["Mean"] = table.mean(axis=1)
        table = table.sort_values("Mean")
    return table


def build_main_quantile_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    table = summary_df.pivot(index="model", columns="task", values="wspl_mean").copy()
    if not table.empty:
        table["Mean"] = table.mean(axis=1)
        table = table.sort_values("Mean")
    return table


def dataframe_to_latex_main_table(
    table_df: pd.DataFrame,
    caption: str,
    label: str,
    metric_name: str,
    lower_is_better: bool = True,
    ndigits: int = 4,
) -> str:
    if table_df.empty:
        return f"% {label}: no data available\n"

    df = table_df.copy()
    cols = list(df.columns)

    styled_cols: Dict[str, List[str]] = {}
    for c in cols:
        styled_cols[c] = _bold_best_and_underline_second(df[c], lower_is_better=lower_is_better, ndigits=ndigits)

    models = [_latex_escape(m) for m in df.index.tolist()]
    col_spec = "l" + "c" * len(cols)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{threeparttable}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Model & " + " & ".join([_latex_escape(c) for c in cols]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for i, model in enumerate(models):
        vals = [styled_cols[c][i] for c in cols]
        lines.append(model + " & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(rf"\footnotesize \item Entries report {metric_name}. Lower values indicate better performance. Best results are shown in bold and second-best results are underlined.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_table4_ablation_csv(ablation_rows: pd.DataFrame) -> pd.DataFrame:
    quant_summary = summarize_quantile_rows(ablation_rows)
    if not quant_summary.empty:
        table = quant_summary.pivot(index="model", columns="task", values="wspl_mean").copy()
    else:
        point_summary = summarize_point_rows(ablation_rows)
        table = point_summary.pivot(index="model", columns="task", values="wrmsse_mean").copy()
    if not table.empty:
        table["Mean"] = table.mean(axis=1)
        table = table.sort_values("Mean")
    return table


def ablation_table_to_latex(table_df: pd.DataFrame, metric_name: str = "WSPL") -> str:
    return dataframe_to_latex_main_table(
        table_df,
        caption="Ablation study on cross-level representation learning.",
        label="tab:ablation",
        metric_name=metric_name,
        lower_is_better=True,
        ndigits=4,
    )


def robustness_table5a_to_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return "% tab:robustness_table5a: no data available\n"

    if "fold" in df.columns and "wrmsse" in df.columns:
        df = (
            df.groupby(["task", "training_origins_used", "model"], as_index=False)
            .agg(wrmsse=("wrmsse", "mean"))
        )

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Robustness to data scarcity.}")
    lines.append(r"\label{tab:robustness_table5a}")
    lines.append(r"\begin{threeparttable}")

    tasks = df["task"].dropna().unique().tolist()
    for t_i, task in enumerate(tasks):
        sub = df.loc[df["task"] == task].copy()
        pivot = sub.pivot(index="training_origins_used", columns="model", values="wrmsse")
        pivot = pivot.sort_index()
        cols = ["AggregateHistGB", "M3_FullSkuTemporalCNN", "DeepSets", "SetTransformer"]
        cols = [c for c in cols if c in pivot.columns] + [c for c in pivot.columns if c not in cols]
        pivot = pivot[cols]

        col_spec = "l" + "c" * len(pivot.columns)
        if t_i > 0:
            lines.append(r"\vspace{1em}")
        lines.append(rf"\textit{{Task: {_latex_escape(task)}}}\\")
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
        lines.append("Training origins used & " + " & ".join([_latex_escape(c) for c in pivot.columns]) + r" \\")
        lines.append(r"\midrule")
        for idx, row in pivot.iterrows():
            vals = [_fmt(v, 4) for v in row.values]
            lines.append(_latex_escape(idx) + " & " + " & ".join(vals) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize \item Entries report WRMSSE. Lower values indicate better performance.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def robustness_table5b_to_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return "% tab:robustness_table5b: no data available\n"

    if "fold" in df.columns and "wrmsse" in df.columns:
        df = (
            df.groupby(["task", "model_capacity"], as_index=False)
            .agg(
                wrmsse=("wrmsse", "mean"),
                wape=("wape", "mean"),
                train_final_loss=("train_final_loss", "mean"),
                overfitting_gap=("overfitting_gap", "mean"),
            )
        )

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sensitivity to model capacity.}")
    lines.append(r"\label{tab:robustness_table5b}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Task & Model capacity & WRMSSE & WAPE & Train final loss & Overfitting gap \\")
    lines.append(r"\midrule")

    show_cols = ["task", "model_capacity", "wrmsse", "wape", "train_final_loss", "overfitting_gap"]
    tmp = df[show_cols].copy()
    for _, row in tmp.iterrows():
        lines.append(
            f"{_latex_escape(row['task'])} & "
            f"{_latex_escape(row['model_capacity'])} & "
            f"{_fmt(row['wrmsse'], 4)} & "
            f"{_fmt(row['wape'], 4)} & "
            f"{_fmt(row['train_final_loss'], 4)} & "
            f"{_fmt(row['overfitting_gap'], 4)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize \item Lower WRMSSE and WAPE are better. A larger overfitting gap indicates a bigger difference between final validation loss and final training loss.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_text(path: str, text: str) -> None:
    ensure_dir(str(Path(path).parent))
    Path(path).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge M5 result CSVs and export paper-ready LaTeX tables.")
    parser.add_argument("--benchmark-rows", type=str, default="", help="CSV from run_m5_full_benchmark.py, e.g. all_rolling_rows.csv")
    parser.add_argument("--proposed-rows", type=str, default="", help="CSV from run_m5_proposed_models.py, e.g. proposed_all_rolling_rows.csv")
    parser.add_argument("--ablation-rows", type=str, default="", help="CSV from run_m5_ablation.py, e.g. all_ablation_rows_rolling.csv")
    parser.add_argument("--table5a-csv", type=str, default="", help="CSV from run_m5_robustness.py, e.g. Table5A_data_scarcity_rows.csv")
    parser.add_argument("--table5b-csv", type=str, default="", help="CSV from run_m5_robustness.py, e.g. Table5B_capacity_rows.csv")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save merged CSVs and LaTeX files.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    benchmark_rows = load_csv(args.benchmark_rows)
    proposed_rows = load_csv(args.proposed_rows)
    ablation_rows = load_csv(args.ablation_rows)
    table5a_df = load_csv(args.table5a_csv)
    table5b_df = load_csv(args.table5b_csv)

    # Tables 2 and 3
    combined_rows = merge_rows(benchmark_rows, proposed_rows)
    combined_rows.to_csv(os.path.join(args.output_dir, "combined_rows.csv"), index=False)

    point_summary = summarize_point_rows(combined_rows)
    point_summary.to_csv(os.path.join(args.output_dir, "point_summary.csv"), index=False)
    point_table = build_main_point_table(point_summary)
    point_table.to_csv(os.path.join(args.output_dir, "Table2_main_point.csv"))

    table2_tex = dataframe_to_latex_main_table(
        point_table,
        caption="Point forecasting results across aggregate levels.",
        label="tab:main_point",
        metric_name="WRMSSE",
        lower_is_better=True,
        ndigits=4,
    )
    write_text(os.path.join(args.output_dir, "Table2_main_point.tex"), table2_tex)

    quant_summary = summarize_quantile_rows(combined_rows)
    quant_summary.to_csv(os.path.join(args.output_dir, "quantile_summary.csv"), index=False)
    if not quant_summary.empty:
        quant_table = build_main_quantile_table(quant_summary)
        quant_table.to_csv(os.path.join(args.output_dir, "Table3_main_quantile.csv"))
        table3_tex = dataframe_to_latex_main_table(
            quant_table,
            caption="Quantile forecasting results across aggregate levels.",
            label="tab:main_quantile",
            metric_name="WSPL",
            lower_is_better=True,
            ndigits=4,
        )
    else:
        table3_tex = "% tab:main_quantile: no quantile rows provided\n"
    write_text(os.path.join(args.output_dir, "Table3_main_quantile.tex"), table3_tex)

    # Table 4
    if ablation_rows is not None:
        ablation_rows.to_csv(os.path.join(args.output_dir, "ablation_rows.csv"), index=False)
        table4_df = build_table4_ablation_csv(ablation_rows)
        table4_df.to_csv(os.path.join(args.output_dir, "Table4_ablation.csv"))
        table4_metric = "WSPL" if "wspl" in ablation_rows.columns and ablation_rows["wspl"].notna().any() else "WRMSSE"
        table4_tex = ablation_table_to_latex(table4_df, metric_name=table4_metric)
    else:
        table4_tex = "% tab:ablation: no ablation rows provided\n"
    write_text(os.path.join(args.output_dir, "Table4_ablation.tex"), table4_tex)

    # Table 5A and 5B
    if table5a_df is not None:
        table5a_df.to_csv(os.path.join(args.output_dir, "Table5A_data_scarcity_rows.csv"), index=False)
        table5a_tex = robustness_table5a_to_latex(table5a_df)
    else:
        table5a_tex = "% tab:robustness_table5a: no robustness scarcity CSV provided\n"
    write_text(os.path.join(args.output_dir, "Table5A_data_scarcity.tex"), table5a_tex)

    if table5b_df is not None:
        table5b_df.to_csv(os.path.join(args.output_dir, "Table5B_capacity_rows.csv"), index=False)
        table5b_tex = robustness_table5b_to_latex(table5b_df)
    else:
        table5b_tex = "% tab:robustness_table5b: no robustness capacity CSV provided\n"
    write_text(os.path.join(args.output_dir, "Table5B_capacity.tex"), table5b_tex)

    manifest = {
        "combined_rows": os.path.join(args.output_dir, "combined_rows.csv"),
        "Table2_main_point_csv": os.path.join(args.output_dir, "Table2_main_point.csv"),
        "Table2_main_point_tex": os.path.join(args.output_dir, "Table2_main_point.tex"),
        "Table3_main_quantile_tex": os.path.join(args.output_dir, "Table3_main_quantile.tex"),
        "Table4_ablation_tex": os.path.join(args.output_dir, "Table4_ablation.tex"),
        "Table5A_data_scarcity_tex": os.path.join(args.output_dir, "Table5A_data_scarcity.tex"),
        "Table5B_capacity_tex": os.path.join(args.output_dir, "Table5B_capacity.tex"),
    }
    write_text(os.path.join(args.output_dir, "manifest.json"), pd.Series(manifest).to_json(indent=2))

    print("Done. Wrote merged CSVs and LaTeX tables to:")
    print(args.output_dir)


if __name__ == "__main__":
    main()