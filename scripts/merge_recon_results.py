from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

MODEL = "Reconcilation"


def append_new_model(recon_csv: Path, target_csv: Path) -> str:
    """Append Reconcilation rows from recon_csv into target_csv. Idempotent."""
    if not recon_csv.exists():
        raise FileNotFoundError(f"Missing recon CSV: {recon_csv}")
    recon = pd.read_csv(recon_csv)
    if recon.empty:
        return "empty recon csv"
    recon = recon.loc[recon["model"].astype(str) == MODEL].copy()
    if recon.empty:
        raise ValueError(f"No {MODEL} rows found in {recon_csv}")
    if target_csv.exists():
        target = pd.read_csv(target_csv)
        n_before = int(len(target))
        if MODEL in set(target["model"].astype(str)):
            return f"already present ({n_before} rows untouched)"
    else:
        target = pd.DataFrame()
        n_before = 0
    merged = pd.concat([target, recon], axis=0, ignore_index=True)
    merged.to_csv(target_csv, index=False)
    return f"appended {len(recon)} rows (was {n_before})"


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Reconcilation benchmark rows into existing experiment CSVs and regenerate stats/tables."
    )
    parser.add_argument("--work-dir", required=True, help="e.g. results/work/10pct")
    parser.add_argument("--mode", required=True, choices=["holdout", "rolling"])
    args = parser.parse_args()

    root = Path(args.work_dir)
    scripts = Path(__file__).resolve().parent
    bench = root / "03_benchmark_baselines"
    recon_dir = bench / "recon"
    mode = args.mode

    print(f"== Merging {MODEL} rows for {root} ({mode}) ==")
    print(f"  rows: {append_new_model(recon_dir / f'all_{mode}_rows.csv', bench / f'all_{mode}_rows.csv')}")
    print(f"  series rows: {append_new_model(recon_dir / f'all_{mode}_series_rows.csv', bench / f'all_{mode}_series_rows.csv')}")

    benchmark_rows = str(bench / f"all_{mode}_rows.csv")
    benchmark_series = str(bench / f"all_{mode}_series_rows.csv")
    proposed_rows = str(root / "04_benchmark_proposed" / f"proposed_all_{mode}_rows.csv")
    proposed_series = str(root / "04_benchmark_proposed" / f"proposed_all_{mode}_series_rows.csv")
    ablation_rows = str(root / "02_ablation" / f"all_ablation_rows_{mode}.csv")
    ablation_series = str(root / "02_ablation" / f"all_ablation_series_rows_{mode}.csv")
    tables_dir = root / "06_paper_tables"
    stats_dir = root / "05_stat_tests"
    tables_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print("== Regenerating paper tables ==")
    run_cmd([
        sys.executable, str(scripts / "make_m5_paper_tables.py"),
        "--benchmark-rows", benchmark_rows,
        "--proposed-rows", proposed_rows,
        "--ablation-rows", ablation_rows,
        "--output-dir", str(tables_dir),
    ])

    print("== Regenerating WSPL stat tests ==")
    run_cmd([
        sys.executable, str(scripts / "run_m5_stat_tests.py"),
        "--benchmark-rows", benchmark_rows,
        "--proposed-rows", proposed_rows,
        "--ablation-rows", ablation_rows,
        "--benchmark-series-rows", benchmark_series,
        "--proposed-series-rows", proposed_series,
        "--ablation-series-rows", ablation_series,
        "--ablation-models", "M0_AggHistOnly,M1_AggHistFutureSummary",
        "--metric", "wspl",
        "--analysis-level", "series",
        "--series-block-cols", "task,agg_id",
        "--output-dir", str(stats_dir),
    ])

    print("== Consistency check ==")
    run_cmd([
        sys.executable, str(scripts / "check_experiment_consistency.py"),
        "--work-dir", str(root), "--mode", mode, "--metric", "wspl",
    ])

    print(f"Done. Merged {MODEL} into {root} and regenerated tables/stats.")


if __name__ == "__main__":
    main()
