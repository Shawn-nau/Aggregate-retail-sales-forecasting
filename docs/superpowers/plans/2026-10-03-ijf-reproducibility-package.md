# IJF Reproducibility Package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the IJF reproducibility package (Package A: `reproducibility_package/` + zip; Package B: `cache_package/`) for the accepted paper "Aggregate retail sales forecasting with child-level set representation learning".

**Architecture:** A small stdlib-only Python packaging module builds Package A by copying curated files from the existing working tree (paper dir, code repo, M5 data, results) into a clean package folder, applying exclusion rules and writing a manifest. Package B is a plain-folder copy of the five preprocessed cache directories. All canonical package-file sources (README, LICENSE, environment files) live inside the code repo under `packaging/package_files/` so they are git-versioned; the assembly scripts copy them into the package.

**Tech Stack:** Python 3 stdlib (`pathlib`, `shutil`, `zipfile`, `argparse`), bash, TeX Live 2022 (`latexmk` + `biber`), git.

## Global Constraints

- Working root: `D:\agentic\aggregate forecasting reproducibility\aggerate last` (NOT a git repo). Git repo: `<root>/aggerate last/scripts_modified_checked_readme` (referred to below as `REPO`). Paper dir: `<root>/aggregate_forecasting_R1`.
- Package A output: `<root>/reproducibility_package/`; zip: `<root>/reproducibility_package_2026-10-03.zip`. Package B output: `<root>/cache_package/` (plain folder, NO archive).
- Main paper results = 100% SKU sample: `results/work/100pct` (holdout) + `results/work/100pct_rolling` (rolling), frac 1.0, seed 42. Hyperparameter tuning is NOT part of the reproduction flow (frozen params in `01_tuning/*/best_params.json`).
- Package A exclusions (never copy): `results/holdout`, `results/rolling`, `results/work/{sanity_debug, work.zip, work_*.tar.gz, v-3.tex, references.bib}`, any `optuna_*.db`, `cache/`, `outputs/`, `.git/`, `.claude/`, `__pycache__/`, `*.pyc`, `m5_experiment_ssh.text*`, `scripts_large_sample_tuning/` and its zip.
- Environment pins and original-machine hardware details are deliberately `[TO CONFIRM]` / `PLACEHOLDER` in the package files until the user provides the pip freeze (spec-approved).
- Commits go to the git repo at `REPO`, following its conventional style (`feat:` / `docs:`).
- Never commit `data/`, `cache/`, or SSH keys (already gitignored in REPO).

---

### Task 1: Packaging module `assemble_package.py` + unit test

**Files:**
- Create: `REPO/packaging/assemble_package.py`
- Test: `REPO/packaging/test_assemble_package.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `CopyItem(src: Path, dst: Path, kind: str)` — dataclass; `kind` is `"file"` or `"dir"`; method `to_manifest_line(package_root: Path) -> str` returns `"<rel-path>\t<bytes>\t<src>"`.
  - `build_copy_plan(repo_dir: Path, paper_dir: Path, package_root: Path) -> List[CopyItem]` — full Package A copy plan.
  - `prune(path: Path) -> bool` — True if a path must be skipped during copy.
  - `copy_items(items: List[CopyItem]) -> None`
  - `write_manifest(items: List[CopyItem], package_root: Path, manifest_path: Path) -> None`
  - `write_git_sha(sha: str, code_dir: Path) -> None` — writes `code_dir/.git_commit_sha.txt`
  - `zip_package(package_root: Path, out_zip: Path) -> None`
  - CLI: `python packaging/assemble_package.py [--clean] [--zip] [--dry-run]` (defaults resolve paths relative to the script: repo_dir = `Path(__file__).resolve().parent.parent`, paper_dir = repo_dir.parent.parent / "aggregate_forecasting_R1", package_root = repo_dir.parent.parent / "reproducibility_package").

- [ ] **Step 1: Write the failing test**

Create `REPO/packaging/test_assemble_package.py` (repo convention: plain asserts, no pytest):

```python
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble_package as ap


def _make_fake_tree():
    tmp = Path(tempfile.mkdtemp(prefix="apkg_test_"))
    repo = tmp / "repo"
    paper = tmp / "aggregate_forecasting_R1"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "run.py").write_text("x=1\n")
    (repo / "scripts" / "__pycache__").mkdir()
    (repo / "scripts" / "__pycache__" / "run.cpython-39.pyc").write_bytes(b"\x00")
    (repo / "scripts" / "skip.pyo").write_bytes(b"\x00")
    (repo / "m5_experiment_ssh.text").write_text("SECRET")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("pass\n")
    (repo / "best_params_all_models.json").write_text("{}")
    (repo / "README.md").write_text("# readme\n")
    (repo / ".gitignore").write_text("data/\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "time_origins.md").write_text("# origins\n")
    (repo / "data").mkdir()
    (repo / "data" / "m5").mkdir(parents=True)
    (repo / "data" / "m5" / "sales_train_evaluation.csv").write_text("a,b\n")
    (repo / "data" / "m5" / "calendar.csv").write_text("c,d\n")
    (repo / "data" / "m5" / "sell_prices.csv").write_text("e,f\n")
    (repo / "data" / "m5" / "sales_train_validation.csv").write_text("g,h\n")
    (repo / "cache").mkdir()
    (repo / "cache" / "100pct").mkdir()
    (repo / "cache" / "100pct" / "base_panel.pkl").write_bytes(b"\x00")
    (repo / "outputs").mkdir()
    work = repo / "results" / "work"
    (work / "100pct").mkdir(parents=True)
    (work / "100pct" / "optuna_DeepSets.db").write_bytes(b"\x00")
    (work / "100pct" / "06_paper_tables").mkdir()
    (work / "100pct" / "06_paper_tables" / "Table2_main_point.tex").write_text("t\n")
    (work / "100pct_rolling").mkdir()
    (work / "10pct").mkdir()
    (work / "10pct_rolling").mkdir()
    (work / "30pct").mkdir()
    (work / "50pct").mkdir()
    (work / "figures").mkdir()
    (work / "figures" / "f.png").write_bytes(b"\x00")
    (work / "sanity_debug").mkdir()
    (work / "v-3.tex").write_text("old\n")
    (work / "work.zip").write_bytes(b"\x00")
    (work / "references.bib").write_text("@misc{x}\n")
    (repo / "results" / "holdout").mkdir()
    (repo / "results" / "holdout" / "stale.tex").write_text("stale\n")
    (repo / "packaging").mkdir()
    (repo / "packaging" / "package_files").mkdir()
    for name in ["README.md", "LICENSE", "environment.yml", "requirements.txt"]:
        (repo / "packaging" / "package_files" / name).write_text("pkg:" + name + "\n")
    (paper).mkdir()
    for name in ["main.pdf", "main.tex", "references.bib", "frame.png",
                 "Figure_cd_diagram_wrmsse_series_holdout.png",
                 "Figure_cd_diagram_wrmsse_series_rolling_foldlevel.png"]:
        (paper / name).write_bytes(b"\x00")
    return tmp, repo, paper


def _clean(tmp):
    shutil.rmtree(tmp, ignore_errors=True)


def test_plan_and_copy_and_zip():
    tmp, repo, paper = _make_fake_tree()
    pkg = tmp / "reproducibility_package"
    try:
        plan = ap.build_copy_plan(repo, paper, pkg)
        dsts = {item.dst for item in plan}
        assert pkg / "paper" / "main.tex" in dsts
        assert pkg / "paper" / "Figure_cd_diagram_wrmsse_series_holdout.png" in dsts
        assert pkg / "code" / "scripts" in dsts
        assert pkg / "code" / "tests" in dsts
        assert pkg / "data" / "m5" / "sales_train_evaluation.csv" in dsts
        assert pkg / "results" / "work" / "100pct" in dsts
        assert pkg / "results" / "work" / "figures" in dsts
        assert pkg / "results" / "work" / "sanity_debug" not in dsts
        assert pkg / "results" / "work" / "v-3.tex" not in dsts
        assert pkg / "results" / "work" / "work.zip" not in dsts
        assert pkg / "results" / "holdout" not in dsts
        assert pkg / "code" / "cache" not in dsts
        assert pkg / "code" / "m5_experiment_ssh.text" not in dsts

        ap.copy_items(plan)
        ap.write_git_sha("abc123", pkg / "code")
        assert (pkg / "code" / ".git_commit_sha.txt").read_text().strip() == "abc123"
        # pruning inside copied dirs
        assert not (pkg / "code" / "scripts" / "__pycache__").exists()
        assert not (pkg / "code" / "scripts" / "skip.pyo").exists()
        assert not (pkg / "results" / "work" / "100pct" / "optuna_DeepSets.db").exists()
        assert (pkg / "results" / "work" / "100pct" / "06_paper_tables" / "Table2_main_point.tex").exists()

        manifest = pkg / "manifest.tsv"
        ap.write_manifest(plan, pkg, manifest)
        lines = manifest.read_text(encoding="utf-8").strip().splitlines()
        assert any(ln.startswith("results/work/100pct\t") for ln in lines)
        assert any("main.tex" in ln for ln in lines)

        z = tmp / "pkg.zip"
        ap.zip_package(pkg, z)
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
        assert "paper/main.tex" in names
        assert "manifest.tsv" in names
        assert "code/.git_commit_sha.txt" in names
        print("test_plan_and_copy_and_zip PASS")
    finally:
        _clean(tmp)


if __name__ == "__main__":
    test_plan_and_copy_and_zip()
    print("ALL TESTS PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "REPO" && python packaging/test_assemble_package.py`
Expected: `ModuleNotFoundError: No module named 'assemble_package'`

- [ ] **Step 3: Write the module**

Create `REPO/packaging/assemble_package.py` with exactly this content:

```python
"""Assemble the IJF reproducibility package (Package A) from the working tree.

Copies a curated set of files from the code repo, the paper directory and the
M5 data directory into a clean package folder, applies exclusion rules, writes
a manifest and (optionally) zips the result. All stdlib-only.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

PAPER_FILES = ["main.pdf", "main.tex", "references.bib", "frame.png"]
CODE_INCLUDE = [
    "scripts", "tests", "best_params_all_models.json",
    "appendix_rolling_folds_point.tex", "appendix_rolling_folds_quantile.tex",
    "docs/time_origins.md", "README.md", ".gitignore",
]
DATA_FILES = ["sales_train_evaluation.csv", "sales_train_validation.csv",
              "calendar.csv", "sell_prices.csv"]
WORK_DIRS = ["100pct", "100pct_rolling", "10pct", "10pct_rolling",
             "30pct", "50pct", "figures"]
PACKAGE_FILES = {
    "README.md": "",
    "LICENSE": "",
    "environment.yml": "environment",
    "requirements.txt": "environment",
}


@dataclass(frozen=True)
class CopyItem:
    src: Path
    dst: Path
    kind: str  # "file" | "dir"

    def to_manifest_line(self, package_root: Path) -> str:
        if self.kind == "dir":
            size = sum(f.stat().st_size for f in self.src.rglob("*") if f.is_file())
        else:
            size = self.src.stat().st_size
        return f"{self.dst.relative_to(package_root)}\t{size}\t{self.src}"


def prune(path: Path) -> bool:
    """Return True if the path must be skipped during copy."""
    name = path.name
    if name == "__pycache__" or name.endswith((".pyc", ".pyo")):
        return True
    if name.startswith("optuna_") and name.endswith(".db"):
        return True
    if name in {"m5_experiment_ssh.text", "m5_experiment_ssh.text.pub"}:
        return True
    return False


def build_copy_plan(repo_dir: Path, paper_dir: Path, package_root: Path) -> List[CopyItem]:
    items: List[CopyItem] = []

    for name in PAPER_FILES + sorted(p.name for p in paper_dir.glob("Figure_cd_diagram_*.png")):
        items.append(CopyItem(paper_dir / name, package_root / "paper" / name, "file"))

    data_dir = repo_dir / "data" / "m5"
    for name in DATA_FILES:
        items.append(CopyItem(data_dir / name, package_root / "data" / "m5" / name, "file"))

    for name in CODE_INCLUDE:
        src = repo_dir / name
        items.append(CopyItem(src, package_root / "code" / name,
                              "dir" if src.is_dir() else "file"))

    work = repo_dir / "results" / "work"
    for name in WORK_DIRS:
        items.append(CopyItem(work / name, package_root / "results" / "work" / name, "dir"))

    for name, sub in PACKAGE_FILES.items():
        src = repo_dir / "packaging" / "package_files" / name
        dst_dir = package_root / sub if sub else package_root
        items.append(CopyItem(src, dst_dir / name, "file"))

    return items


def copy_items(items: List[CopyItem]) -> None:
    for item in items:
        item.dst.parent.mkdir(parents=True, exist_ok=True)
        if item.kind == "dir":
            shutil.copytree(item.src, item.dst, dirs_exist_ok=True)
            bad_paths = sorted((p for p in item.dst.rglob("*") if prune(p)),
                               key=lambda p: len(p.parts), reverse=True)
            for bad in bad_paths:
                shutil.rmtree(bad) if bad.is_dir() else bad.unlink()
        else:
            shutil.copy2(item.src, item.dst)


def write_manifest(items: List[CopyItem], package_root: Path, manifest_path: Path) -> None:
    header = "package_path\tsize_bytes\tsource_path\n"
    manifest_path.write_text(
        header + "\n".join(item.to_manifest_line(package_root) for item in items) + "\n",
        encoding="utf-8")


def write_git_sha(sha: str, code_dir: Path) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / ".git_commit_sha.txt").write_text(sha + "\n", encoding="utf-8")


def zip_package(package_root: Path, out_zip: Path) -> None:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(package_root.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(package_root))


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    root = repo_dir.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default=str(repo_dir))
    parser.add_argument("--paper-dir", default=str(root / "aggregate_forecasting_R1"))
    parser.add_argument("--package-dir", default=str(root / "reproducibility_package"))
    parser.add_argument("--zip", action="store_true", help="Create the zip archive after assembling.")
    parser.add_argument("--clean", action="store_true", help="Delete the package dir first.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without copying.")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    paper_dir = Path(args.paper_dir)
    package_root = Path(args.package_dir)

    if args.clean and package_root.exists():
        shutil.rmtree(package_root)

    plan = build_copy_plan(repo_dir, paper_dir, package_root)
    missing = [i.src for i in plan if not i.src.exists()]
    if missing:
        print("ERROR: missing sources:", *missing, sep="\n  ", file=sys.stderr)
        return 1

    if args.dry_run:
        for item in plan:
            print(item.to_manifest_line(package_root))
        return 0

    copy_items(plan)
    sha = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if sha:
        write_git_sha(sha, package_root / "code")
    write_manifest(plan, package_root, package_root / "manifest.tsv")
    print(f"Package assembled at {package_root}: {len(plan)} items (see manifest.tsv)")

    if args.zip:
        out_zip = package_root.parent / f"{package_root.name}_2026-10-03.zip"
        zip_package(package_root, out_zip)
        print(f"Zip written: {out_zip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "REPO" && python packaging/test_assemble_package.py`
Expected: `test_plan_and_copy_and_zip PASS` then `ALL TESTS PASS`

- [ ] **Step 5: Commit**

```bash
cd "REPO"
git add packaging/assemble_package.py packaging/test_assemble_package.py
git commit -m "feat: add reproducibility package assembly script with unit test"
```

---

### Task 2: Package README source

**Files:**
- Create: `REPO/packaging/package_files/README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: the canonical Package A README; Task 4 copies it to `reproducibility_package/README.md` via the plan built in Task 1 (`PACKAGE_FILES`).

- [ ] **Step 1: Write the README**

Create `REPO/packaging/package_files/README.md` with exactly this content:

````markdown
# Reproducibility Package

## Aggregate retail sales forecasting with child-level set representation learning

**Paper:** *Aggregate retail sales forecasting with child-level set representation learning*, International Journal of Forecasting (accepted; R1 revision, August 2026).

- **Assembly date:** 2026-10-03
- **Paper authors:** Shaohui Ma, Shengkai Wang — School of Business, Nanjing Audit University, No. 86 Yushan West Road, Nanjing, China, 211815
- **Contact for reproducibility issues:** shaohui.ma@nau.edu.cn
- **Code licence:** MIT (see `LICENSE`). Data licence: see the Data section.

---

## Special setup requirements — read this first

- **GPU is required.** The proposed neural models (Gated Pooling / SkuTemporalCNN, DeepSets, Set Transformer) are PyTorch models trained on GPU. An NVIDIA GPU with a CUDA-enabled PyTorch build is required to re-run the main experiments. The external benchmark models (sklearn) are CPU-only.
- **Disk space.** Regenerating the preprocessed sample cache for the 100% runs needs roughly 105 GB (25 GB holdout + 80 GB rolling). Use the supplementary `cache_package` (see below) to skip cache building entirely.
- **RAM.** Building the cache constructs a dense item–store panel (about 59 million rows × 26 features) plus per-task tensors in memory. The original runs used a cloud GPU instance ([hardware details to confirm]).
- **Long runs.** Run experiments inside `tmux`/`screen` over SSH so they survive disconnects.
- **Thread limiting.** Set these before every run to avoid CPU oversubscription:

  ```bash
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  ```

- **Hyperparameter selection is not part of the reproduction flow.** All experiments below use the pre-selected hyperparameters shipped in `results/work/*/01_tuning/`. The Optuna tuning script (`code/scripts/tune_m5_proposed_models.py`) is included for completeness but is not needed to reproduce the paper's results.

## Repository structure

```
.
├── README.md                  <- this file
├── LICENSE                    <- MIT licence for the code
├── manifest.tsv               <- every file in this package: path, size, source path
├── environment/               <- pinned environment files (conda + pip)
├── paper/                     <- paper source and figures
│   ├── main.pdf               <- compiled paper
│   ├── main.tex               <- LaTeX source (tables are inline in this file)
│   ├── references.bib
│   ├── frame.png              <- architecture figure (hand-made, not generated by code)
│   └── Figure_cd_diagram_*.png  <- the 4 critical-difference diagrams used by main.tex
├── code/                      <- snapshot of the experiment code repository
│   ├── scripts/               <- all experiment scripts
│   ├── tests/test_ols_reconciled.py
│   ├── best_params_all_models.json   <- 10% warm-start hyperparameters
│   ├── appendix_rolling_folds_*.tex  <- generated rolling appendix tables (sources pasted into main.tex)
│   ├── docs/time_origins.md   <- exact train/valid/test origin dates for both modes
│   ├── README.md              <- the original implementation guide (detailed workflow + troubleshooting)
│   └── .git_commit_sha.txt    <- git commit of the code snapshot
├── data/m5/                   <- M5 competition data (raw CSVs)
└── results/work/              <- pre-generated experiment outputs (see below)
    ├── 100pct/                <- MAIN holdout results (100% SKU sample)
    ├── 100pct_rolling/        <- MAIN rolling-origin results (100% SKU sample)
    ├── 10pct/ 10pct_rolling/ 30pct/ 50pct/   <- robustness runs (sample-fraction analyses)
    └── figures/               <- CD-diagram PNGs with paper-consistent model names
```

The preprocessed sample cache is **not included** (about 128 GB in total). It is provided separately as the supplementary `cache_package` folder, or it can be regenerated from the raw data (see "Cache regeneration" below).

## Computing environment

- **Operating system:** Linux (the original runs were executed on a cloud GPU instance; the local paths in the run logs are `/root/autodl-tmp/m5`). [Hardware details: CPU/RAM/GPU model — to confirm by authors]
- **Language:** Python 3.11 (conda environment named `m5_forecast`).
- **Environment recreation:**

  ```bash
  conda env create -f environment/environment.yml
  conda activate m5_forecast
  # or, with pip:
  pip install -r environment/requirements.txt
  ```

  Install PyTorch with the CUDA build for your GPU using the command generated by the official selector at https://pytorch.org/get-started/locally/ (for recent Linux GPU environments this is `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX`).

- **Key packages:** numpy, pandas, scipy, scikit-learn, matplotlib, optuna, wandb, threadpoolctl, tqdm, pytorch (CUDA). Exact pinned versions: see `environment/` files (the pins correspond to the `pip freeze` of the original machine [to confirm by authors]).
- **Licence:** code is MIT (`LICENSE`); the M5 data is distributed under the Kaggle terms of the M5 competition (see Data section).

## Data

The experiments use the **M5 forecasting competition dataset** (Makridakis et al., 2022; Kaggle "M5 Forecasting - Accuracy", https://www.kaggle.com/competitions/m5-forecasting-accuracy). It is public and freely available after a free Kaggle registration.

- **Files included in `data/m5/`:** `sales_train_evaluation.csv` (main file, 3,049 products × 1,941 days), `calendar.csv` (dates, SNAP and event flags), `sell_prices.csv` (weekly SKU–store prices), `sales_train_validation.csv` (fallback if the evaluation file is absent).
- **Preprocessing pipeline** (`code/scripts/prepare_m5_experiments.py`): the wide sales table is melted into a long item–store panel; calendar features (weekday, month, year, SNAP, events) and price features (sell price, price-change ratio) are merged; a store-SKU sample is drawn with `--store-sku-sample-frac` and seed 42; aggregate units are formed per task (`store_dept`, `store_cat`, `state_dept`); and dense per-aggregate samples are cached (cache version `v5_child_targets_bottomup_global_boosting`).

## Intermediary datasets

Pre-generated phase outputs are included under `results/work/`. Each directory maps to its generating script:

| Directory | Generated by |
|---|---|
| `results/work/100pct/01_tuning/` | `tune_m5_proposed_models.py` — **frozen hyperparameters**; consumed as input by the reproduction flow below |
| `results/work/*/02_ablation/` | `run_m5_ablation.py` |
| `results/work/*/03_benchmark_baselines/` | `run_m5_full_benchmark.py` |
| `results/work/*/04_benchmark_proposed/` | `run_m5_proposed_models.py` |
| `results/work/*/05_stat_tests/` | `run_m5_stat_tests.py` + `generate_cd_diagrams_100pct.py` |
| `results/work/*/06_paper_tables/` | `make_m5_paper_tables.py` |
| `results/work/figures/` | `generate_cd_diagrams_100pct.py` |
| `results/work/10pct/robustness_5seeds/` | `run_m5_robustness.py` (via `run_robustness_5seeds.sh`) |

The **sample cache** (not included) sits between the raw data and these phase outputs: it is generated by `code/scripts/warm_m5_cache.py` and consumed by every experiment script. If you use the supplementary `cache_package`, you can skip this step entirely.

## Which code produces which outputs

All table numbers in `paper/main.tex` are inlined LaTeX (there are no `\input` commands). The mapping below tells you which script produces each table/figure and where the pre-generated version lives in this package.

| Paper element (label in main.tex) | Generating script(s) | Pre-generated output in this package |
|---|---|---|
| Table 2 — point forecasting (tab:main_point) | `code/scripts/make_m5_paper_tables.py` | `results/work/100pct/06_paper_tables/Table2_main_point.tex` |
| Table 3 — quantile forecasting (tab:main_quantile) | same | `results/work/100pct/06_paper_tables/Table3_main_quantile.tex` |
| Ablation rows (M0/M1) inside Tables 2–3 | `run_m5_ablation.py` → merged by `make_m5_paper_tables.py` | `results/work/100pct/02_ablation/` + `Table4_ablation.tex` (standalone ablation table, generated but not used as a separate table in main.tex) |
| Figure 1 — architecture (fig:architecture_frame) | hand-made (not generated) | `paper/frame.png` |
| Figure 2 — point CD diagram (fig:point_statistical) | `code/scripts/generate_cd_diagrams_100pct.py` (reads `results/work/100pct/*_series_rows.csv`) | `paper/Figure_cd_diagram_wrmsse_series_holdout.png`; underlying stats in `results/work/100pct/05_stat_tests/` |
| Figure 3 — quantile CD diagram (fig:quantile_statistical) | same | `paper/Figure_cd_diagram_mean_spl_series_holdout.png`; stats in `results/work/100pct/05_stat_tests/` |
| Rolling-origin tables (tab:rolling_point, tab:rolling_quantile) | `make_m5_paper_tables.py` (rolling mode) | `results/work/100pct_rolling/06_paper_tables/Table2_main_point.tex`, `Table3_main_quantile.tex` |
| Rolling CD figures (fig:rolling_point_statistical, fig:rolling_quantile_statistical) | `generate_cd_diagrams_100pct.py` (rolling tag) | `paper/Figure_cd_diagram_*_series_rolling_foldlevel.png`; stats in `results/work/100pct_rolling/05_stat_tests/` |
| Per-origin rolling appendix tables (tab:appendix_rolling_folds_point, tab:appendix_rolling_folds_quantile) | `code/scripts/build_rolling_fold_tables.py` | `code/appendix_rolling_folds_point.tex`, `code/appendix_rolling_folds_quantile.tex` |
| Sample-fraction summary tables (tab:summary_point_sample_frac, tab:summary_quantile_sample_frac) | `make_m5_paper_tables.py` run at each fraction | `results/work/{30pct,50pct,100pct}/06_paper_tables/Table2_main_point.tex`, `Table3_main_quantile.tex` |
| Sample-fraction appendix tables (tab:appendix_point_sample_fractions_combined, tab:appendix_quantile_sample_fractions_combined) | `code/scripts/build_robustness_tables.py` | fraction work dirs + `results/work/10pct/robustness_5seeds/` |
| Training-time table (tab:compute_time) | compiled manually from run logs | `results/work/100pct{,_rolling}/run_all.log` |
| Hyperparameter tables (tab:appendix_tuning_ranges, tab:appendix_tuning_configs) | from tuning artifacts (no script) | `results/work/100pct/01_tuning/<model>/tuning_run_config.json` + `best_params.json` |

Note on the CD figures: the PNGs shipped in `paper/` are the exported copies used in the manuscript (filenames end in `_holdout` / `_rolling_foldlevel`). Re-running `generate_cd_diagrams_100pct.py` reproduces the same statistics and equivalent diagrams from the included result rows; re-exported PNGs may differ cosmetically (font rendering) from the copies in `paper/`.

## Hardware and expected runtime

The original runs were executed on a cloud GPU instance ([CPU/RAM/GPU model — to confirm by authors]). Approximate runtimes, taken from the run logs (`results/work/*/run_all.log`) and log file dates:

| Step | Approx. runtime (original machine) |
|---|---|
| Cache build, 100% holdout (25 GB) | ~2 hours (panel 1:48 min + ~1:20 min per task + sample pickling) |
| Cache build, 100% rolling (80 GB, holdout + rolling folds) | several hours |
| Full holdout experiments (ablation + benchmarks + proposed models + stats + tables, using frozen hyperparameters) | about 1 day |
| Full rolling experiments (same phases, rolling mode) | about 3–4 days |
| Benchmark-only phases (sklearn) | minutes per task |
| Stats + paper tables | minutes |

The tuning phase (excluded from this flow) is visible in `results/work/100pct/run_all.log` and took several hours per model on the original machine.

## Reproduction flows

Paths below assume you are in `code/` with `data/` and `results/` from this package alongside, on a Linux machine with the conda environment activated.

### Flow A — Main results (100% SKU sample) using the cache package

1. Copy the caches from the supplementary `cache_package` into `code/cache/` (or point `--cache-dir` at them directly):
   ```bash
   cp -r ../cache_package/100pct cache/
   cp -r ../cache_package/100pct_rolling cache/
   ```
2. Create fresh work dirs and copy the frozen hyperparameters into them (this replaces the tuning phase):
   ```bash
   mkdir -p work/holdout/01_tuning work/rolling/01_tuning
   cp -r ../results/work/100pct/01_tuning/* work/holdout/01_tuning/
   cp -r ../results/work/100pct_rolling/01_tuning/* work/rolling/01_tuning/
   ```
3. Holdout run:
   ```bash
   python scripts/run_paper_experiment_plan.py \
     --data-dir data/m5 --work-dir work/holdout --cache-dir cache/100pct \
     --mode holdout --tasks store_dept,store_cat,state_dept \
     --proposed-models M3_FullSkuTemporalCNN,DeepSets,SetTransformer \
     --ablation-params-model M3_FullSkuTemporalCNN \
     --phases ablation,benchmark,stats,tables \
     --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
     --store-sku-sample-frac 1.0 --store-sku-sample-seed 42 \
     --final-epochs 24 --final-patience 6 --stats-metric wspl
   ```
4. Rolling run (same command with `--mode rolling --work-dir work/rolling --cache-dir cache/100pct_rolling`).
5. Consistency check after each run:
   ```bash
   python scripts/check_experiment_consistency.py --work-dir work/holdout --mode holdout --metric wspl
   python scripts/check_experiment_consistency.py --work-dir work/rolling --mode rolling --metric wspl
   ```
6. Compare your fresh `work/holdout` against the pre-generated `results/work/100pct/` (and `work/rolling` against `results/work/100pct_rolling/`): the paper tables (`06_paper_tables/Table2_main_point.tex` etc.) and the stat tables in `05_stat_tests/` must agree.

### Flow B — Rebuild the cache from raw data (skip the cache package)

```bash
# 100% holdout cache (~2 h)
python scripts/warm_m5_cache.py --data-dir data/m5 --cache-dir cache/100pct \
  --tasks store_dept,store_cat,state_dept --holdout \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac 1.0 --store-sku-sample-seed 42
# 100% rolling cache (holdout + rolling folds; several hours, ~80 GB)
python scripts/warm_m5_cache.py --data-dir data/m5 --cache-dir cache/100pct_rolling \
  --tasks store_dept,store_cat,state_dept --holdout --rolling \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac 1.0 --store-sku-sample-seed 42
```
Then continue with Flow A from step 2.

### Flow C — Results-only verification (no retraining)

Cross-check the shipped results against the paper without running anything:

- Spot-check Table 2: `results/work/100pct/06_paper_tables/Table2_main_point.csv` — e.g. Set Transformer mean WRMSSE = 0.6897 (matches paper Table 2).
- Spot-check Table 3: `results/work/100pct/06_paper_tables/Table3_main_quantile.csv`.
- Statistical tables: `results/work/100pct{,_rolling}/05_stat_tests/Table_stats_*.csv` (Friedman ranks, Nemenyi pairwise p-values).
- CD diagrams: `results/work/figures/Figure_cd_diagram_*_{holdout,rolling}.png` and `paper/Figure_cd_diagram_*.png`.

### Robustness runs (10%, 30%, 50% SKU samples)

Same pattern as Flow A at the given fraction (cache dirs `cache/{10pct,30pct,50pct}` in the cache package; frozen hyperparameters from `results/work/<fraction>/01_tuning/`; `--store-sku-sample-frac 0.10|0.30|0.50`). The 10% rolling robustness run has no shipped cache (rebuild with `--holdout --rolling --store-sku-sample-frac 0.10`).

### Unit test

```bash
cd code && python tests/test_ols_reconciled.py   # expected output: ALL TESTS PASS
```

## Licence

- **Code:** MIT — see `LICENSE`.
- **Data:** M5 competition dataset, obtained from Kaggle (https://www.kaggle.com/competitions/m5-forecasting-accuracy); redistributed here for reproducibility purposes under the Kaggle terms of the competition.

## IJF reproducibility checklist

| Requirement | Where addressed |
|---|---|
| Assembly date stated | Header of this README |
| Authors and contact information | Header of this README |
| Repository structure described | "Repository structure" |
| Computing environment, languages, packages, versions | "Computing environment" + `environment/` |
| Environment recreation instructions | "Computing environment" |
| Licences stated | "Licence" |
| Data: what it is, where it comes from, preprocessing | "Data" |
| Intermediary datasets identified + generating scripts | "Intermediary datasets" |
| Every table/figure mapped to its script | "Which code produces which outputs" |
| Hardware used and expected runtimes | "Hardware and expected runtime" |
| Special setup requirements flagged | "Special setup requirements" at the top |
````

- [ ] **Step 2: Verify the README renders and has all sections**

Run: `cd "REPO" && python - <<'PY'
text = open("packaging/package_files/README.md", encoding="utf-8").read()
for section in ["Assembly date", "Special setup requirements", "Repository structure",
                "Computing environment", "## Data", "Intermediary datasets",
                "Which code produces which outputs", "Hardware and expected runtime",
                "Reproduction flows", "IJF reproducibility checklist"]:
    assert section in text, f"missing section: {section}"
print("README sections OK")
PY`
Expected: `README sections OK`

- [ ] **Step 3: Commit**

```bash
cd "REPO"
git add packaging/package_files/README.md
git commit -m "docs: add IJF reproducibility package README"
```

---

### Task 3: Environment files + LICENSE

**Files:**
- Create: `REPO/packaging/package_files/environment.yml`
- Create: `REPO/packaging/package_files/requirements.txt`
- Create: `REPO/packaging/package_files/LICENSE`

**Interfaces:**
- Consumes: nothing.
- Produces: package files copied by Task 1's `PACKAGE_FILES` map into `reproducibility_package/environment/` and `reproducibility_package/LICENSE`.

- [ ] **Step 1: Write environment.yml**

Create `REPO/packaging/package_files/environment.yml`:

```yaml
name: m5_forecast
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
      # PLACEHOLDER — replace the unpinned list below with the pinned
      # `pip freeze` output from the machine that produced the paper results
      # (to be provided by the authors).
      - numpy
      - pandas
      - scipy
      - scikit-learn
      - matplotlib
      - optuna
      - wandb
      - threadpoolctl
      - tqdm
      # PyTorch: install the CUDA build via the official selector
      # (https://pytorch.org/get-started/locally/) instead of the plain wheel,
      # e.g.: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
```

- [ ] **Step 2: Write requirements.txt**

Create `REPO/packaging/package_files/requirements.txt`:

```
# PLACEHOLDER — replace the unpinned list below with the pinned `pip freeze`
# output from the machine that produced the paper results (to be provided by
# the authors).
numpy
pandas
scipy
scikit-learn
matplotlib
optuna
wandb
threadpoolctl
tqdm
# torch: install the CUDA wheel for your GPU from https://pytorch.org/get-started/locally/
```

- [ ] **Step 3: Write LICENSE (MIT)**

Create `REPO/packaging/package_files/LICENSE`:

```
MIT License

Copyright (c) 2026 Shaohui Ma, Shengkai Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Verify**

Run: `cd "REPO" && ls -la packaging/package_files/`
Expected: `environment.yml`, `requirements.txt`, `LICENSE`, `README.md` all present and non-empty.

- [ ] **Step 5: Commit**

```bash
cd "REPO"
git add packaging/package_files/environment.yml packaging/package_files/requirements.txt packaging/package_files/LICENSE
git commit -m "docs: add environment files and MIT license for reproducibility package"
```

---

### Task 4: Compile the paper, assemble Package A, verify

**Files:**
- No new files (fixes only if checks fail).

**Interfaces:**
- Consumes: Task 1 module + Task 2/3 package files.
- Produces: `<root>/reproducibility_package/` (complete Package A, with `paper/main.pdf` compiled fresh).

- [ ] **Step 1: Compile the paper PDF**

Run: `cd "<root>/aggregate_forecasting_R1" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: ends with `Latexmk: All targets (main.pdf) are up-to-date` or `Output written on main.pdf`; `main.pdf` exists. (First run invokes pdflatex + biber automatically; may take a few minutes.)

- [ ] **Step 2: Assemble Package A**

Run: `cd "REPO" && python packaging/assemble_package.py --clean`
Expected: `Package assembled at ...reproducibility_package: N items (see manifest.tsv)`

- [ ] **Step 3: Verify key files exist**

Run: `cd "REPO" && python - <<'PY'
from pathlib import Path
pkg = Path("../../reproducibility_package").resolve()
required = [
    "README.md", "LICENSE", "manifest.tsv",
    "environment/environment.yml", "environment/requirements.txt",
    "paper/main.pdf", "paper/main.tex", "paper/references.bib", "paper/frame.png",
    "code/scripts/run_paper_experiment_plan.py",
    "code/scripts/warm_m5_cache.py",
    "code/tests/test_ols_reconciled.py",
    "code/best_params_all_models.json",
    "code/.git_commit_sha.txt",
    "code/docs/time_origins.md", "code/README.md", "code/.gitignore",
    "data/m5/sales_train_evaluation.csv", "data/m5/calendar.csv",
    "data/m5/sell_prices.csv",
    "results/work/100pct/06_paper_tables/Table2_main_point.tex",
    "results/work/100pct/06_paper_tables/Table3_main_quantile.tex",
    "results/work/100pct/06_paper_tables/Table4_ablation.tex",
    "results/work/100pct_rolling/06_paper_tables/Table2_main_point.tex",
    "results/work/figures/Figure_cd_diagram_mean_spl_series_holdout.png",
    "results/work/figures/Figure_cd_diagram_wrmsse_series_rolling.png",
]
missing = [p for p in required if not (pkg / p).exists()]
assert not missing, f"missing: {missing}"
print(f"All {len(required)} required files present")
PY`
Expected: `All 26 required files present`

- [ ] **Step 4: Verify excluded artifacts did not leak in**

Run: `cd "<root>" && find reproducibility_package -name "optuna_*.db" -o -name "m5_experiment_ssh*" -o -name "*.pyc" -o -name "__pycache__" | head; ls reproducibility_package/results/; ls reproducibility_package/code/`
Expected: the `find` prints nothing; `results/` contains only `work/`; `code/` contains no `cache/`, no `outputs/`, no `.git/`.

- [ ] **Step 5: Verify frozen hyperparameters for the fixed-parameter flow**

Run: `cd "REPO" && python - <<'PY'
from pathlib import Path
pkg = Path("../../reproducibility_package").resolve()
for wd in ["results/work/100pct", "results/work/100pct_rolling"]:
    for model in ["M3_FullSkuTemporalCNN", "DeepSets", "SetTransformer"]:
        p = pkg / wd / "01_tuning" / model / "best_params.json"
        assert p.exists(), f"missing frozen params: {p}"
print("Frozen hyperparameters complete for holdout and rolling")
PY`
Expected: `Frozen hyperparameters complete for holdout and rolling`

- [ ] **Step 6: Attempt the repo unit test (document outcome)**

Run: `cd "REPO" && python tests/test_ols_reconciled.py`
Expected: either `ALL TESTS PASS`, or a `ModuleNotFoundError` for a scientific stack package on this local machine (Python 3.8.5 without the experiment environment) — in that case record "unit test could not run locally; it targets the package environment" in the session notes, no code change needed.

- [ ] **Step 7: Commit any fixes**

Only if Steps 3–5 required fixing Task 1–3 files:
```bash
cd "REPO" && git add -A packaging/ && git commit -m "fix: packaging adjustments found during assembly verification"
```

---

### Task 5: Cache package (Package B) script + README + copy run

**Files:**
- Create: `REPO/packaging/assemble_cache_package.py`
- Create: `REPO/packaging/package_files/cache_package_README.md`
- Test: `REPO/packaging/test_assemble_cache_package.py`

**Interfaces:**
- Consumes: `CopyItem`, `copy_items`, `write_manifest`, `prune` from Task 1 (`import assemble_package as ap`).
- Produces: `CACHE_DIRS = ["10pct", "30pct", "50pct", "100pct", "100pct_rolling"]`; `build_cache_plan(repo_dir, cache_package_root) -> List[CopyItem]`; CLI `python packaging/assemble_cache_package.py [--clean] [--dry-run]` writing `<root>/cache_package/`.

- [ ] **Step 1: Write the failing test**

Create `REPO/packaging/test_assemble_cache_package.py`:

```python
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble_cache_package as acp


def test_cache_plan():
    tmp = Path(tempfile.mkdtemp(prefix="acp_test_"))
    repo = tmp / "repo"
    try:
        for name in ["10pct", "30pct", "50pct", "100pct", "100pct_rolling", "m5_cache"]:
            (repo / "cache" / name).mkdir(parents=True)
            (repo / "cache" / name / "base_panel.pkl").write_bytes(b"\x00")
        (repo / "packaging").mkdir()
        (repo / "packaging" / "package_files").mkdir()
        (repo / "packaging" / "package_files" / "cache_package_README.md").write_text("# cache readme\n")
        cache_root = tmp / "cache_package"
        plan = acp.build_cache_plan(repo, cache_root)
        dsts = {str(item.dst) for item in plan}
        assert str(cache_root / "100pct_rolling") in dsts
        assert str(cache_root / "50pct") in dsts
        assert str(cache_root / "m5_cache") not in dsts
        assert str(cache_root / "README.md") in dsts
        acp.copy_items(plan)
        assert (cache_root / "100pct" / "base_panel.pkl").exists()
        assert (cache_root / "README.md").exists()
        print("test_cache_plan PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_cache_plan()
    print("ALL TESTS PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "REPO" && python packaging/test_assemble_cache_package.py`
Expected: `ModuleNotFoundError: No module named 'assemble_cache_package'`

- [ ] **Step 3: Write the cache-package script**

Create `REPO/packaging/assemble_cache_package.py`:

```python
"""Assemble the supplementary cache package (Package B): plain-folder copy of
the preprocessed sample caches. Copying ~128 GB takes a while — run it in the
background."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

import assemble_package as ap

CACHE_DIRS = ["10pct", "30pct", "50pct", "100pct", "100pct_rolling"]


def build_cache_plan(repo_dir: Path, cache_package_root: Path) -> List[ap.CopyItem]:
    items: List[ap.CopyItem] = []
    for name in CACHE_DIRS:
        items.append(ap.CopyItem(repo_dir / "cache" / name, cache_package_root / name, "dir"))
    items.append(ap.CopyItem(
        repo_dir / "packaging" / "package_files" / "cache_package_README.md",
        cache_package_root / "README.md", "file"))
    return items


def copy_items(items: List[ap.CopyItem]) -> None:
    for item in items:
        item.dst.parent.mkdir(parents=True, exist_ok=True)
        if item.kind == "dir":
            shutil.copytree(item.src, item.dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item.src, item.dst)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    root = repo_dir.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default=str(repo_dir))
    parser.add_argument("--cache-package-dir", default=str(root / "cache_package"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    cache_package_root = Path(args.cache_package_dir)

    if args.clean and cache_package_root.exists():
        shutil.rmtree(cache_package_root)

    plan = build_cache_plan(repo_dir, cache_package_root)
    missing = [i.src for i in plan if not i.src.exists()]
    if missing:
        print("ERROR: missing sources:", *missing, sep="\n  ", file=sys.stderr)
        return 1

    if args.dry_run:
        for item in plan:
            print(item.to_manifest_line(cache_package_root))
        return 0

    copy_items(plan)
    ap.write_manifest(plan, cache_package_root, cache_package_root / "manifest.tsv")
    print(f"Cache package assembled at {cache_package_root}: {len(plan)} items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "REPO" && python packaging/test_assemble_cache_package.py`
Expected: `test_cache_plan PASS` then `ALL TESTS PASS`

- [ ] **Step 5: Write the cache package README**

Create `REPO/packaging/package_files/cache_package_README.md`:

````markdown
# Cache package — preprocessed M5 experiment samples

Supplementary package for the reproducibility kit of "Aggregate retail sales forecasting with child-level set representation learning". It contains the preprocessed sample caches so the reproduction run can **skip cache building entirely** (the slowest non-GPU step).

Assembly date: 2026-10-03. Cache version: `v5_child_targets_bottomup_global_boosting`. All caches were built with `--store-sku-sample-seed 42`.

## Contents

| Directory | Size | SKU fraction | Modes included | Enables |
|---|---|---|---|---|
| `100pct/` | 25 GB | 1.00 | holdout only | MAIN holdout results (paper Tables 2–3, Figures 2–3) |
| `100pct_rolling/` | 80 GB | 1.00 | holdout + rolling folds 1–5 | MAIN rolling results (paper rolling tables/figures) |
| `10pct/` | 2.5 GB | 0.10 | holdout only | 10% robustness run |
| `30pct/` | 7.4 GB | 0.30 | holdout only | 30% robustness run |
| `50pct/` | 13 GB | 0.50 | holdout only | 50% robustness run |

Each directory contains `base_panel.pkl` plus `task__state_dept/`, `task__store_cat/`, `task__store_dept/` with the per-task sample pickles (`final_train_samples.pkl`, `internal_valid_samples.pkl`, `test_samples.pkl`, and — in `100pct_rolling/` — `rolling_fold_1..5_{train,valid}_samples.pkl`).

## How each directory was generated

`warm_m5_cache.py` in the main package's `code/scripts/`, run with `--cache-dir` set to that directory:

```bash
# 100pct (holdout)
python scripts/warm_m5_cache.py --data-dir data/m5 --cache-dir cache/100pct \
  --tasks store_dept,store_cat,state_dept --holdout \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac 1.0 --store-sku-sample-seed 42

# 100pct_rolling (holdout + rolling)
python scripts/warm_m5_cache.py --data-dir data/m5 --cache-dir cache/100pct_rolling \
  --tasks store_dept,store_cat,state_dept --holdout --rolling \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac 1.0 --store-sku-sample-seed 42
```

The 10/30/50pct directories use the same holdout command with `--store-sku-sample-frac 0.10|0.30|0.50`.

## How to use

1. Copy the needed directory next to the main package's code (or anywhere on a fast disk), e.g. `cp -r cache_package/100pct reproducibility_package/code/cache/`.
2. Pass it via `--cache-dir` with the matching `--store-sku-sample-frac` and `--store-sku-sample-seed 42` in every experiment command (see the main package README, Flow A).
3. Do **not** rename files inside the directories, and do **not** mix directories with different SKU fractions — each experiment must use the cache built for its exact fraction.

## Transferring to a GPU server

These are large binary files; use rsync over SSH (resumable):

```bash
rsync -avP cache_package/ user@SERVER:~/m5/cache/
```

## Notes

- No cache is included for the 10% rolling robustness run — rebuild it with `--holdout --rolling --store-sku-sample-frac 0.10` (see main README, Flow B).
- The cache is regenerable from the raw M5 data in the main package (`data/m5/`) with the commands above; keeping the shipped copy avoids ~2 hours (holdout) to several hours (rolling) of preprocessing per fraction.
````

- [ ] **Step 6: Verify the cache README sections**

Run: `cd "REPO" && python - <<'PY'
text = open("packaging/package_files/cache_package_README.md", encoding="utf-8").read()
for section in ["## Contents", "How each directory was generated", "## How to use", "## Transferring"]:
    assert section in text, f"missing: {section}"
print("cache README sections OK")
PY`
Expected: `cache README sections OK`

- [ ] **Step 7: Commit**

```bash
cd "REPO"
git add packaging/assemble_cache_package.py packaging/test_assemble_cache_package.py packaging/package_files/cache_package_README.md
git commit -m "feat: add cache package assembly script and README"
```

- [ ] **Step 8: Run the real cache copy (background, ~30–60+ min)**

Run: `cd "REPO" && python packaging/assemble_cache_package.py`
Expected: after completion (check later), `Package assembled at ...cache_package: 6 items` and `du -sh <root>/cache_package/*` shows ~128 GB total across the five dirs. Run this step in the background while Task 6 proceeds.

---

### Task 6: Zip Package A

**Files:**
- No new files.

**Interfaces:**
- Consumes: Task 4's `reproducibility_package/`.
- Produces: `<root>/reproducibility_package_2026-10-03.zip`.

- [ ] **Step 1: Create the zip**

Run: `cd "REPO" && python packaging/assemble_package.py --zip`
Expected: `Package assembled at ...reproducibility_package: N items (see manifest.tsv)` followed by `Zip written: ...reproducibility_package_2026-10-03.zip` (deflating ~700 MB takes a few minutes; allow up to 10 minutes).

- [ ] **Step 2: Verify the zip contents**

Run: `cd "REPO" && python - <<'PY'
import zipfile
z = zipfile.ZipFile("../../reproducibility_package_2026-10-03.zip")
names = z.namelist()
for required in ["README.md", "LICENSE", "paper/main.pdf", "code/scripts/run_paper_experiment_plan.py",
                 "data/m5/sales_train_evaluation.csv", "results/work/100pct/06_paper_tables/Table2_main_point.tex",
                 "manifest.tsv"]:
    assert required in names, f"missing in zip: {required}"
bad = [n for n in names if "optuna_" in n or "ssh" in n or "holdout/" in n]
assert not bad, f"excluded artifacts in zip: {bad[:5]}"
print(f"zip OK: {len(names)} entries")
PY`
Expected: `zip OK: <large number> entries`

- [ ] **Step 3: Confirm Package B copy finished**

Run: `ls "<root>/cache_package/" && du -sh "<root>/cache_package"/*`
Expected: `README.md manifest.tsv 10pct 30pct 50pct 100pct 100pct_rolling` with sizes ≈ 2.5G / 7.4G / 13G / 25G / 80G.

- [ ] **Step 4: Final report (no commit)**

Report to the user: package paths, sizes, zip name, verification results, and the two open items awaiting their input (pip freeze for pinned environment files; original-machine hardware details).
