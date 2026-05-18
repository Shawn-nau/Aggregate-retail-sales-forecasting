"""
Build appendix and summary LaTeX tables for robustness across SKU sample fractions.

Generates:
- 6 appendix tables (point/quantile × 10%/30%/50%)
- 2 summary tables (point/quantile) showing Mean only across sample fractions
"""
from pathlib import Path
import pandas as pd
import numpy as np

WORK = Path(__file__).resolve().parent.parent / "results" / "work"
OUTPUT = WORK / "v-3.tex"  # We'll print LaTeX, not modify directly

PAPER_NAMES = {
    "SetTransformer": "Set Transformer",
    "SetTransformer_Q": "Set Transformer",
    "DeepSets": "DeepSets",
    "DeepSets_Q": "DeepSets",
    "M3_FullSkuTemporalCNN": "Gated Pooling",
    "M3_FullSkuTemporalCNN_Q": "Gated Pooling",
    "M0_AggHistOnly": "AggHistOnly",
    "M1_AggHistFutureSummary": "AggHist Child-summary",
    "BottomUpGlobalHistGB": "Bottom-up Global HistGB",
    "AggregateHistGB": "Aggregate HistGB",
    "AggregateElasticNet": "Aggregate Elastic Net",
    "ChildSummaryHistGB": "Child-summary HistGB",
    "ChildSummaryElasticNet": "Child-summary Elastic Net",
    "SeasonalNaive": "Seasonal Naive",
}

TABLE_ORDER_POINT = [
    "SetTransformer", "DeepSets", "M3_FullSkuTemporalCNN",
    "BottomUpGlobalHistGB", "M0_AggHistOnly", "M1_AggHistFutureSummary",
    "ChildSummaryHistGB", "AggregateHistGB",
    "AggregateElasticNet", "ChildSummaryElasticNet", "SeasonalNaive",
]
TABLE_ORDER_QUANTILE = [
    "SetTransformer_Q", "DeepSets_Q", "M3_FullSkuTemporalCNN_Q",
    "BottomUpGlobalHistGB", "M0_AggHistOnly", "M1_AggHistFutureSummary",
    "AggregateElasticNet", "ChildSummaryElasticNet",
    "AggregateHistGB", "ChildSummaryHistGB", "SeasonalNaive",
]

def fmt(x, ndigits=4):
    if pd.isna(x):
        return ""
    return f"{x:.{ndigits}f}"

def build_model_rows(data_dict, metric, order):
    """data_dict: {model: {task: value, ...}} with 'Mean' computed."""
    # Create a lookup for means
    means = {m: d['Mean'] for m, d in data_dict.items()}
    sorted_models = sorted(order, key=lambda m: means.get(m, 999), reverse=False)

    # Determine bold/underline per column
    cols = ['state_dept', 'store_cat', 'store_dept', 'Mean']
    annotations = {col: {} for col in cols}
    for col in cols:
        vals = [(m, data_dict[m][col]) for m in sorted_models if col in data_dict[m]]
        vals.sort(key=lambda x: x[1])
        if len(vals) >= 1:
            annotations[col][vals[0][0]] = 'textbf'
        if len(vals) >= 2:
            annotations[col][vals[1][0]] = 'underline'

    rows = []
    for model in order:
        if model not in data_dict:
            continue
        name = PAPER_NAMES.get(model, model)
        vals = []
        for col in cols:
            v = data_dict[model].get(col, np.nan)
            s = fmt(v)
            ann = annotations[col].get(model, '')
            if ann == 'textbf':
                s = f"\\textbf{{{s}}}"
            elif ann == 'underline':
                s = f"\\underline{{{s}}}"
            vals.append(s)
        rows.append(f"      {name:<30} & {vals[0]:<18} & {vals[1]:<18} & {vals[2]:<18} & {vals[3]:<18} \\\\")
    return rows

def build_table(rows, caption, label, note_text):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(rf"  \caption{{{caption}}}")
    lines.append(rf"  \label{{{label}}}")
    lines.append(r"  \begin{threeparttable}")
    lines.append(r"    \begin{tabular}{lcccc}")
    lines.append(r"      \toprule")
    lines.append(r"      Model                     & state$\times$dept  & store$\times$cat   & store$\times$dept  & Mean               \\")
    lines.append(r"      \midrule")
    for r in rows:
        lines.append(r)
    lines.append(r"      \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \begin{tablenotes}")
    lines.append(rf"      \footnotesize \item {note_text}")
    lines.append(r"    \end{tablenotes}")
    lines.append(r"  \end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def load_ablation(pct):
    """Return {model: {state_dept/...: {wrmsse, wspl}}} for M0 and M1."""
    abl = pd.read_csv(WORK / pct / "02_ablation" / "all_ablation_rows_holdout.csv")
    result = {}
    for model in ["M0_AggHistOnly", "M1_AggHistFutureSummary"]:
        mdf = abl[abl['model'] == model]
        d = {}
        # wrmsse and wspl are in separate rows — aggregate by task
        for task in mdf['task'].unique():
            tdf = mdf[mdf['task'] == task]
            wrmsse_vals = tdf['wrmsse'].dropna()
            wspl_vals = tdf['wspl'].dropna()
            d[task] = {
                'wrmsse': float(wrmsse_vals.iloc[0]) if len(wrmsse_vals) > 0 else np.nan,
                'wspl': float(wspl_vals.iloc[0]) if len(wspl_vals) > 0 else np.nan,
            }
        wrmsse_all = [v['wrmsse'] for v in d.values() if not pd.isna(v['wrmsse'])]
        wspl_all = [v['wspl'] for v in d.values() if not pd.isna(v['wspl'])]
        result[model] = {
            'state_dept': d.get('state_dept', {'wrmsse': np.nan, 'wspl': np.nan}),
            'store_cat': d.get('store_cat', {'wrmsse': np.nan, 'wspl': np.nan}),
            'store_dept': d.get('store_dept', {'wrmsse': np.nan, 'wspl': np.nan}),
            'wrmsse_mean': np.mean(wrmsse_all) if wrmsse_all else np.nan,
            'wspl_mean': np.mean(wspl_all) if wspl_all else np.nan,
        }
    return result

# ── Load all data ────────────────────────────────────────────────────────────

all_data = {}
for pct in ['10pct', '30pct', '50pct', '100pct']:
    pt = pd.read_csv(WORK / pct / "06_paper_tables" / "Table2_main_point.csv")
    qt = pd.read_csv(WORK / pct / "06_paper_tables" / "Table3_main_quantile.csv")
    abl = load_ablation(pct)
    all_data[pct] = {'point': pt, 'quantile': qt, 'ablation': abl}

# ── Generate appendix tables ─────────────────────────────────────────────────

appendix_tables = {}
for pct, label_pct in [('10pct', '10\\%'), ('30pct', '30\\%'), ('50pct', '50\\%')]:
    pt = all_data[pct]['point']
    qt = all_data[pct]['quantile']
    abl = all_data[pct]['ablation']

    # Build point data dict
    pt_data = {}
    for _, row in pt.iterrows():
        model = row['model']
        pt_data[model] = {
            'state_dept': row['state_dept'],
            'store_cat': row['store_cat'],
            'store_dept': row['store_dept'],
            'Mean': row['Mean'],
        }
    # Add ablation M0 and M1
    for m in ['M0_AggHistOnly', 'M1_AggHistFutureSummary']:
        if m in abl:
            pt_data[m] = {
                'state_dept': abl[m]['state_dept']['wrmsse'],
                'store_cat': abl[m]['store_cat']['wrmsse'],
                'store_dept': abl[m]['store_dept']['wrmsse'],
                'Mean': abl[m]['wrmsse_mean'],
            }

    pt_rows = build_model_rows(pt_data, 'wrmsse', TABLE_ORDER_POINT)
    pt_table = build_table(
        pt_rows,
        f"Point forecasting results with {label_pct} SKU sample.",
        f"tab:appendix_point_{pct}",
        f"Entries report WRMSSE with {label_pct} SKU--store series sampled. Lower values indicate better performance. Best results are shown in bold and second-best results are underlined.",
    )
    appendix_tables[f'point_{pct}'] = pt_table

    # Build quantile data dict
    qt_data = {}
    for _, row in qt.iterrows():
        model = row['model']
        qt_data[model] = {
            'state_dept': row['state_dept'],
            'store_cat': row['store_cat'],
            'store_dept': row['store_dept'],
            'Mean': row['Mean'],
        }
    for m in ['M0_AggHistOnly', 'M1_AggHistFutureSummary']:
        if m in abl:
            qt_data[m] = {
                'state_dept': abl[m]['state_dept']['wspl'],
                'store_cat': abl[m]['store_cat']['wspl'],
                'store_dept': abl[m]['store_dept']['wspl'],
                'Mean': abl[m]['wspl_mean'],
            }

    qt_rows = build_model_rows(qt_data, 'wspl', TABLE_ORDER_QUANTILE)
    qt_table = build_table(
        qt_rows,
        f"Probabilistic forecasting results with {label_pct} SKU sample.",
        f"tab:appendix_quantile_{pct}",
        f"Entries report WSPL with {label_pct} SKU--store series sampled. Lower values indicate better performance. Best results are shown in bold and second-best results are underlined.",
    )
    appendix_tables[f'quantile_{pct}'] = qt_table

# Print appendix tables
print("=" * 80)
print("APPENDIX TABLES")
print("=" * 80)
for key in ['point_10pct', 'point_30pct', 'point_50pct',
            'quantile_10pct', 'quantile_30pct', 'quantile_50pct']:
    print(f"\n% --- {key} ---")
    print(appendix_tables[key])

# ── Build summary tables (Mean only) ─────────────────────────────────────────

SUMMARY_MODELS_POINT = [
    "SetTransformer", "DeepSets", "M3_FullSkuTemporalCNN",
    "BottomUpGlobalHistGB", "M0_AggHistOnly", "M1_AggHistFutureSummary",
    "ChildSummaryHistGB", "AggregateHistGB",
    "SeasonalNaive", "AggregateElasticNet", "ChildSummaryElasticNet",
]
SUMMARY_MODELS_QUANTILE = [
    "SetTransformer_Q", "DeepSets_Q", "M3_FullSkuTemporalCNN_Q",
    "BottomUpGlobalHistGB", "M0_AggHistOnly", "M1_AggHistFutureSummary",
    "AggregateHistGB", "ChildSummaryHistGB",
    "SeasonalNaive", "AggregateElasticNet", "ChildSummaryElasticNet",
]

PCTS = ['10pct', '30pct', '50pct', '100pct']
PCT_LABELS = ['10\\%', '30\\%', '50\\%', '100\\%']

def get_mean(pct, model, metric_type):
    """Get overall Mean for a model at a given sample fraction."""
    if model in ['M0_AggHistOnly', 'M1_AggHistFutureSummary']:
        abl_data = all_data[pct]['ablation']
        if model in abl_data:
            if metric_type == 'wrmsse':
                return abl_data[model]['wrmsse_mean']
            else:
                return abl_data[model]['wspl_mean']
        return np.nan

    table_key = 'point' if metric_type == 'wrmsse' else 'quantile'
    df = all_data[pct][table_key]
    match = df[df['model'] == model]
    if len(match) > 0:
        return match.iloc[0]['Mean']
    return np.nan

# Point summary
print("\n" + "=" * 80)
print("POINT SUMMARY TABLE")
print("=" * 80)

pt_summary_rows = []
for model in SUMMARY_MODELS_POINT:
    name = PAPER_NAMES.get(model, model)
    vals = []
    for pct in PCTS:
        m = get_mean(pct, model, 'wrmsse')
        vals.append(fmt(m))
    pt_summary_rows.append(f"      {name:<30} & {' & '.join(vals)} \\\\")

pt_summary = []
pt_summary.append(r"\begin{table}[H]")
pt_summary.append(r"  \centering")
pt_summary.append(r"  \caption{Point forecasting results (mean WRMSSE) across SKU sample fractions.}")
pt_summary.append(r"  \label{tab:summary_point_sample_frac}")
pt_summary.append(r"  \begin{threeparttable}")
pt_summary.append(r"    \begin{tabular}{lcccc}")
pt_summary.append(r"      \toprule")
pt_summary.append(r"      Model                     & 10\%                & 30\%                & 50\%                & 100\%               \\")
pt_summary.append(r"      \midrule")
for r in pt_summary_rows:
    pt_summary.append(r)
pt_summary.append(r"      \bottomrule")
pt_summary.append(r"    \end{tabular}")
pt_summary.append(r"    \begin{tablenotes}")
pt_summary.append(r"      \footnotesize \item Entries report the overall mean WRMSSE across the three aggregate forecasting tasks (state$\times$department, store$\times$category, store$\times$department). Lower values indicate better performance. Each column corresponds to a different random sampling fraction of the bottom-level SKU--store series.")
pt_summary.append(r"    \end{tablenotes}")
pt_summary.append(r"  \end{threeparttable}")
pt_summary.append(r"\end{table}")
print("\n".join(pt_summary))

# Quantile summary
print("\n" + "=" * 80)
print("QUANTILE SUMMARY TABLE")
print("=" * 80)

qt_summary_rows = []
for model in SUMMARY_MODELS_QUANTILE:
    name = PAPER_NAMES.get(model, model)
    vals = []
    for pct in PCTS:
        m = get_mean(pct, model, 'wspl')
        vals.append(fmt(m))
    qt_summary_rows.append(f"      {name:<30} & {' & '.join(vals)} \\\\")

qt_summary = []
qt_summary.append(r"\begin{table}[H]")
qt_summary.append(r"  \centering")
qt_summary.append(r"  \caption{Probabilistic forecasting results (mean WSPL) across SKU sample fractions.}")
qt_summary.append(r"  \label{tab:summary_quantile_sample_frac}")
qt_summary.append(r"  \begin{threeparttable}")
qt_summary.append(r"    \begin{tabular}{lcccc}")
qt_summary.append(r"      \toprule")
qt_summary.append(r"      Model                     & 10\%                & 30\%                & 50\%                & 100\%               \\")
qt_summary.append(r"      \midrule")
for r in qt_summary_rows:
    qt_summary.append(r)
qt_summary.append(r"      \bottomrule")
qt_summary.append(r"    \end{tabular}")
qt_summary.append(r"    \begin{tablenotes}")
qt_summary.append(r"      \footnotesize \item Entries report the overall mean WSPL across the three aggregate forecasting tasks (state$\times$department, store$\times$category, store$\times$department). Lower values indicate better performance. Each column corresponds to a different random sampling fraction of the bottom-level SKU--store series.")
qt_summary.append(r"    \end{tablenotes}")
qt_summary.append(r"  \end{threeparttable}")
qt_summary.append(r"\end{table}")
print("\n".join(qt_summary))
