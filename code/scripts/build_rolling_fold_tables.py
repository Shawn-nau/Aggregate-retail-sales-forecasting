"""
Build per-fold rolling forecast tables for the appendix.

Reads 100pct_rolling fold-level data and generates two comprehensive
LaTeX tables: one for point (WRMSSE) and one for quantile (WSPL),
showing per-fold per-task per-model results.
"""
from pathlib import Path
import pandas as pd
import numpy as np

WORK = Path(__file__).resolve().parent.parent / "results" / "work"
ROLLING = WORK / "100pct_rolling"

PAPER_NAMES = {
    "SetTransformer": "Set Transformer",
    "SetTransformer_Q": "Set Transformer",
    "DeepSets": "DeepSets",
    "DeepSets_Q": "DeepSets",
    "M3_FullSkuTemporalCNN": "Gated Pooling",
    "M3_FullSkuTemporalCNN_Q": "Gated Pooling",
    "M0_AggHistOnly": "AggHistOnly NN",
    "M1_AggHistFutureSummary": "Child-summary NN",
    "BottomUpGlobalHistGB": "Bottom-up Global HGB",
    "AggregateHistGB": "Aggregate HGB",
    "AggregateElasticNet": "Aggregate Elastic Net",
    "ChildSummaryHistGB": "Child-summary HGB",
    "ChildSummaryElasticNet": "Child-summary Elastic Net",
    "SeasonalNaive": "Aggregate Seasonal Naive",
    "Reconcilation": "Reconciled HGB",
}

MODEL_ORDER_POINT = [
    "SetTransformer", "DeepSets", "M3_FullSkuTemporalCNN",
    "BottomUpGlobalHistGB", "M0_AggHistOnly", "M1_AggHistFutureSummary",
    "ChildSummaryHistGB", "AggregateHistGB", "Reconcilation",
    "SeasonalNaive", "AggregateElasticNet", "ChildSummaryElasticNet",
]
MODEL_ORDER_QUANTILE = [
    "SetTransformer_Q", "DeepSets_Q", "M3_FullSkuTemporalCNN_Q",
    "BottomUpGlobalHistGB", "M0_AggHistOnly", "M1_AggHistFutureSummary",
    "AggregateHistGB", "ChildSummaryHistGB",
    "SeasonalNaive", "AggregateElasticNet", "ChildSummaryElasticNet",
]

TASK_ORDER = ["state_dept", "store_cat", "store_dept"]
TASK_LABELS = {
    "state_dept": r"state$\times$dept",
    "store_cat": r"store$\times$cat",
    "store_dept": r"store$\times$dept",
}

def fmt(x, ndigits=4):
    if pd.isna(x) or x is None:
        return ""
    return f"{x:.{ndigits}f}"

def load_fold_data(csv_path, metric_col):
    """Load fold-level data and pivot: model × task → {fold: value}.
    Returns dict: {model: {task: {fold: value, 'Mean': mean_across_folds}}}"""
    df = pd.read_csv(csv_path)
    result = {}
    for model in df['model'].unique():
        mdf = df[df['model'] == model]
        d = {}
        for task in TASK_ORDER:
            tdf = mdf[mdf['task'] == task]
            vals = tdf[metric_col].dropna()
            fold_vals = {}
            for _, row in tdf.iterrows():
                fold = int(row['fold'])
                v = row[metric_col]
                if not pd.isna(v):
                    fold_vals[fold] = float(v)
            mean_val = np.mean(list(fold_vals.values())) if fold_vals else np.nan
            d[task] = {'folds': fold_vals, 'Mean': mean_val}
            # Overall mean across tasks
        all_vals = []
        for task in TASK_ORDER:
            if task in d:
                all_vals.extend(list(d[task]['folds'].values()))
        result[model] = d
        result[model]['_overall_mean'] = np.mean(all_vals) if all_vals else np.nan
    return result

# Load all data
bench_pt = load_fold_data(ROLLING / "03_benchmark_baselines" / "all_rolling_rows.csv", "wrmsse")
bench_qt = load_fold_data(ROLLING / "03_benchmark_baselines" / "all_rolling_rows.csv", "wspl")

prop_pt = load_fold_data(ROLLING / "04_benchmark_proposed" / "proposed_all_rolling_rows.csv", "wrmsse")
prop_qt = load_fold_data(ROLLING / "04_benchmark_proposed" / "proposed_all_rolling_rows.csv", "wspl")

abl_pt = load_fold_data(ROLLING / "02_ablation" / "all_ablation_rows_rolling.csv", "wrmsse")
abl_qt = load_fold_data(ROLLING / "02_ablation" / "all_ablation_rows_rolling.csv", "wspl")

# Merge all
pt_data = {**bench_pt, **prop_pt}
for m in ["M0_AggHistOnly", "M1_AggHistFutureSummary"]:
    if m in abl_pt:
        pt_data[m] = abl_pt[m]

qt_data = {**bench_qt, **prop_qt}
for m in ["M0_AggHistOnly", "M1_AggHistFutureSummary"]:
    if m in abl_qt:
        qt_data[m] = abl_qt[m]

FOLDS = [1, 2, 3, 4, 5]

def build_fold_table(data, metric_name, model_order):
    """Build LaTeX table with per-fold per-task per-model results.
    Format: Model | Task | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Mean"""
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(rf"  \caption{{{metric_name} results by rolling forecast origin.}}")
    label = "tab:appendix_rolling_folds_" + ("point" if "WRMSSE" in metric_name else "quantile")
    lines.append(rf"  \label{{{label}}}")
    lines.append(r"  \begin{threeparttable}")
    lines.append(r"    \footnotesize")
    lines.append(r"    \begin{tabular}{llccccc|c}")
    lines.append(r"      \toprule")
    lines.append(r"      \textbf{Model} & \textbf{Task} & \textbf{Fold 1} & \textbf{Fold 2} & \textbf{Fold 3} & \textbf{Fold 4} & \textbf{Fold 5} & \textbf{Mean} \\")
    lines.append(r"      \midrule")

    for mi, model in enumerate(model_order):
        if model not in data:
            continue
        name = PAPER_NAMES.get(model, model)
        d = data[model]
        for ti, task in enumerate(TASK_ORDER):
            task_label = TASK_LABELS[task]
            if task in d:
                fold_vals = [fmt(d[task]['folds'].get(f, np.nan)) for f in FOLDS]
                mean_val = fmt(d[task]['Mean'])
            else:
                fold_vals = [""] * 5
                mean_val = ""
            if ti == 0:
                row = f"      {name:<28} & {task_label:<22} & " + " & ".join(f"{v:>8}" for v in fold_vals) + f" & {mean_val:>8} \\\\"
            else:
                row = f"      {'':<28} & {task_label:<22} & " + " & ".join(f"{v:>8}" for v in fold_vals) + f" & {mean_val:>8} \\\\"
            lines.append(row)
        # Overall mean row for this model
        overall = fmt(d.get('_overall_mean', np.nan))
        lines.append(f"      {'':<28} & {{\\bf Overall}} & & & & & & {{\\bf {overall}}} \\\\")
        if mi < len([m for m in model_order if m in data]) - 1:
            lines.append(r"      \addlinespace")

    lines.append(r"      \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \begin{tablenotes}")
    lines.append(r"      \footnotesize \item Entries report " + metric_name.split("(")[0].strip() +
                 r" for each forecast origin (fold). Lower values indicate better performance. The Mean column reports the average across the five rolling origins.")
    lines.append(r"    \end{tablenotes}")
    lines.append(r"  \end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

# Generate tables
pt_table = build_fold_table(pt_data, "WRMSSE (point forecasting)", MODEL_ORDER_POINT)
qt_table = build_fold_table(qt_data, "WSPL (probabilistic forecasting)", MODEL_ORDER_QUANTILE)

# Output to files
with open("appendix_rolling_folds_point.tex", "w") as f:
    f.write(pt_table)
with open("appendix_rolling_folds_quantile.tex", "w") as f:
    f.write(qt_table)

print("Point table saved to appendix_rolling_folds_point.tex")
print("Quantile table saved to appendix_rolling_folds_quantile.tex")
print()

# Also print preview
for model in MODEL_ORDER_POINT[:3]:
    if model in pt_data:
        name = PAPER_NAMES.get(model, model)
        print(f"{name}:")
        d = pt_data[model]
        for task in TASK_ORDER:
            if task in d:
                vals = [f"{d[task]['folds'].get(f, np.nan):.4f}" for f in FOLDS]
                print(f"  {task}: {' | '.join(vals)}  mean={d[task]['Mean']:.4f}")
        print(f"  overall_mean={d['_overall_mean']:.4f}")
        print()
