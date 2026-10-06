# IJF Reproducibility Package — Design

Date: 2026-10-03
Status: approved design (all decisions confirmed by user)

## Goal

Build the reproducibility package for the IJF-accepted paper "Aggregate retail sales forecasting with child-level set representation learning" (Shaohui Ma, Shengkai Wang; R1, August 2026), following the IJF reproducibility guide (`Reproducibility in the International Journal of Forecasting.html`).

Two deliverables, per user decisions:

- **Package A** — `reproducibility_package/`: the main kit (README, LICENSE, environment files, paper source + PDF, code snapshot, M5 data, pre-generated results). Delivered as folder + zip archive.
- **Package B** — `cache_package/`: all preprocessed sample caches (~128G) so reviewers can skip cache building. Delivered as a plain folder with its own README (no archive).

Key user decisions baked in:

1. **Main results = 100% SKU sample** (not 10%): `results/work/100pct` (holdout) and `results/work/100pct_rolling` (rolling) are the paper's main results. Verified: paper Table 2 values (Set Transformer mean WRMSSE 0.6897 etc.) match `results/work/100pct/06_paper_tables/Table2_main_point.csv` exactly.
2. **Hyperparameter selection is out of the reproduction flow**: frozen hyperparameters from the included `01_tuning/*/best_params.json` files replace the tuning phase.
3. **Package B includes all caches** to speed up reproduction.
4. Stale `results/holdout` + `results/rolling` (10pct-era copies that contradict the paper) are **excluded** from Package A.
5. MIT license; pip freeze will be provided by the user later → placeholder pins now.
6. Cache package distributed as a **plain folder** (no tars); rsync instructions in its README.

## Verified facts the design relies on

- Paper: `aggregate_forecasting_R1/main.tex`, tables hard-coded inline (no `\input`), biblatex/biber APA, figures: `frame.png` (hand-made) + 4 CD-diagram PNGs (`*_holdout.png`, `*_rolling_foldlevel.png`).
- `results/work/100pct/` and `results/work/100pct_rolling/` contain the full phase outputs 00_sanity → 06_paper_tables, plus `run_all.log` (runtime evidence) and `optuna_*.db` (tuning artifacts, excluded).
- Orchestrator `run_paper_experiment_plan.py` reads `<work-dir>/01_tuning/<model>/best_params.json` when the `tune` phase is skipped (ablation: line ~267; benchmark proposed: line ~292), so the fixed-parameter flow is: create fresh work dir → copy the included per-model best_params JSONs into `01_tuning/<model>/` → run `--phases ablation,benchmark,stats,tables`. (`best_params_all_models.json` in the work dir is only *written* by the tune phase — not needed in the fixed-param flow; the repo-root `best_params_all_models.json` is the 10pct warm-start file for robustness-fraction retuning.)
- CD figures: `generate_cd_diagrams_100pct.py` regenerates the 100pct holdout + rolling CD diagrams (paper-consistent model names) from the series rows in `results/work/100pct{,_rolling}/03_benchmark_baselines` etc., writing tagged PNGs into `results/work/figures/` (verified: byte-identical to the untagged PNGs in `100pct{,_rolling}/05_stat_tests`). The paper's PNG copies (suffixed `_holdout` / `_rolling_foldlevel`) are exported copies of these figures; the underlying stat CSVs are in `results/work/100pct{,_rolling}/05_stat_tests` (user confirmed the rolling figure's source is `100pct_rolling/05_stat_tests`). The paper PNGs are not byte-identical to the repo PNGs (re-export with different names); the README maps figures via the generating script + data, and ships the paper's own PNGs in `paper/`.
- Cache layout: per-fraction directories, each = `base_panel.pkl` + `task__{state_dept,store_cat,store_dept}/` with sample pickles; cache version `v5_child_targets_bottomup_global_boosting`; seed 42; `100pct_rolling` additionally has `rolling_fold_1..5_{train,valid}_samples.pkl`. Sizes: 10pct 2.5G, 30pct 7.4G, 50pct 13G, 100pct 25G (holdout-only), 100pct_rolling 80G (holdout + rolling). `cache/m5_cache/` (1.3G, store_dept-only demo) excluded.
- No local cache exists for the 10pct rolling robustness run (results exist, cache doesn't) → rebuild-only, documented.
- Local toolchain: TeX Live 2022 (pdflatex, biber, latexmk) available; local Python is 3.8.5 without optuna/torch → the experiment pipeline cannot run locally; the package targets the Linux GPU environment described in the code README.
- The code repo (`aggerate last/scripts_modified_checked_readme/`) is a git repo; the top-level working dir is not.

## Package A structure

```
reproducibility_package/
├── README.md                     # guide-compliant package readme
├── LICENSE                       # MIT
├── environment/
│   ├── environment.yml           # conda, python=3.11, placeholder pins (pending user pip freeze)
│   └── requirements.txt          # pip, placeholder pins
├── paper/
│   ├── main.pdf                  # compiled locally (TeX Live 2022)
│   ├── main.tex
│   ├── references.bib
│   ├── frame.png
│   └── Figure_cd_diagram_*.png   # the 4 CD PNGs used by main.tex
├── code/                         # snapshot of the code repo
│   ├── scripts/                  # all 26 scripts
│   ├── tests/test_ols_reconciled.py
│   ├── best_params_all_models.json
│   ├── appendix_rolling_folds_point.tex   # generated appendix-table sources
│   ├── appendix_rolling_folds_quantile.tex
│   ├── docs/time_origins.md
│   ├── README.md                 # original implementation guide, verbatim
│   └── .gitignore
├── data/m5/                      # sales_train_evaluation.csv, sales_train_validation.csv,
│                                 # calendar.csv, sell_prices.csv (425M)
└── results/
    └── work/                     # pre-generated outputs, authoritative
        ├── 100pct/               # MAIN holdout results (excl. optuna_*.db)
        ├── 100pct_rolling/       # MAIN rolling results (excl. optuna_*.db)
        ├── 10pct/                # robustness (incl. robustness_5seeds)
        ├── 10pct_rolling/
        ├── 30pct/
        ├── 50pct/
        └── figures/              # tagged CD PNGs from generate_cd_diagrams_100pct.py
```

Excluded from Package A (documented in README): `results/holdout` + `results/rolling` (stale 10pct-era), `results/work/{sanity_debug, work.zip, work_*.tar.gz, v-3.tex, references.bib}`, `optuna_*.db`, `cache/` (all of it — see Package B), `outputs/`, `.git/`, `.claude/`, `__pycache__/`, SSH keys (`m5_experiment_ssh.text*`), `scripts_large_sample_tuning/` (+ zip). Git commit SHA of the code snapshot is recorded in the README.

## Package A README outline (per the IJF guide checklist)

1. Title, assembly date (2026-10-03), paper authors, contact (shaohui.ma@nau.edu.cn)
2. **Special setup requirements near the top**: NVIDIA GPU + CUDA PyTorch for the neural models (benchmarks are CPU-only); the 100% runs need ~105G of cache space + large RAM; tmux/SSH for long runs
3. Repository structure (tree above)
4. Computing environment: Linux GPU box (exact CPU/GPU/RAM as [TO CONFIRM] placeholders), Python 3.11, pinned packages (placeholder pending pip freeze), recreation instructions (conda env create / pip install -r)
5. Data: M5 dataset — what it is, Kaggle provenance (`m5-forecasting-accuracy`, cite `makridakis2022m5bg`), formats, preprocessing summary (`read_m5_raw` → store-SKU panel → task samples), included in `data/m5/`
6. Intermediary datasets: `results/work/*` phase outputs mapped to generating scripts; cache NOT included — Package B or regeneration via `warm_m5_cache.py` (per-fraction commands + sizes); 10pct-rolling cache is rebuild-only
7. **Which code produces which outputs**: mapping table for every paper table/figure → script → command → runtime (runtimes parsed from `run_all.log` where available)
8. Hardware and expected runtime per major step
9. Reproduction flows:
   - (a) Main 100% reproduction with Package B: place cache dirs → copy frozen hyperparameters into a fresh work dir (`01_tuning/<model>/best_params.json` for the 3 models, from `results/work/100pct{,_rolling}/01_tuning/`) → run `run_paper_experiment_plan.py --phases ablation,benchmark,stats,tables --store-sku-sample-frac 1.0` for holdout and rolling → `check_experiment_consistency.py` → compare against included `results/work/100pct{,_rolling}`
   - (b) Cache regeneration from raw data (skip Package B): `warm_m5_cache.py` commands per fraction
   - (c) Results-only verification (no retraining): inspect included results and cross-check against the paper's tables/figures
10. Robustness reproduction (10/30/50pct) — brief, same pattern at the given fraction
11. Licenses: MIT (code), M5 data terms (Kaggle competition data, public)
12. IJF self-check table (the guide's final checklist, all items checked)

## Package B structure

```
cache_package/
├── README.md
├── 100pct/            # 25G  holdout samples, frac 1.0 — REQUIRED for main holdout results
├── 100pct_rolling/    # 80G  holdout + rolling folds 1–5, frac 1.0 — REQUIRED for main rolling results
├── 10pct/             # 2.5G robustness
├── 30pct/             # 7.4G robustness
└── 50pct/             # 13G robustness
```

Package B README: inventory table (dir, size, fraction, holdout/rolling, which paper results it enables), the exact `warm_m5_cache.py` command that generated each dir, placement instructions (`--cache-dir` per dir with matching `--store-sku-sample-frac` and `--store-sku-sample-seed 42`), cache version note (`v5_child_targets_bottomup_global_boosting`), rsync/scp transfer instructions, and the note that no cache exists for the 10pct rolling robustness run.

## Assembly

- `assemble_package.py` at the working root: copies the curated set into `reproducibility_package/`, prints a manifest (source → destination, sizes), rerunnable (`--clean` flag).
- `assemble_cache_package.py`: copies the 5 cache dirs into `cache_package/` (run in background; est. 30–60 min).
- Zip: `reproducibility_package_2026-10-03.zip` created after assembly.
- Verification before finalizing:
  1. Compile the paper locally (latexmk -pdf) → `paper/main.pdf` placed in the package.
  2. Manifest check: every path referenced in the package README exists in the package.
  3. Run `tests/test_ols_reconciled.py` if the local Python allows; otherwise note the limitation in the README.
  4. Frozen-hyperparameter check: `01_tuning/<model>/best_params.json` exists for each of the 3 models in both `results/work/100pct` and `results/work/100pct_rolling`.
  5. Confirm no excluded artifacts (SSH keys, optuna DBs, stale dirs) leaked into the package.

## Out of scope

- No tuning workflow in the README's main flow (frozen hyperparameters only; `tune_m5_proposed_models.py` remains in the code snapshot but is marked as outside the reproduction flow).
- Environment pins remain placeholders until the user provides the pip freeze.
- Package B is not archived.
- Local end-to-end pipeline runs (no GPU locally); verification is limited to the items above.
