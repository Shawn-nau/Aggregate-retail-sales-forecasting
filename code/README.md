# M5 Cross-Level Retail Forecasting Experiments: Implementation Guide

This package runs the M5 aggregate forecasting experiments for point and probabilistic forecasting. It supports two evaluation modes:

- `holdout`: one final strict test evaluation.
- `rolling`: expanding-window rolling-origin validation/evaluation.

The code has been checked so that `holdout` and `rolling` are consistently handled across benchmark models, proposed neural models, ablation models, and statistical tests.

---

## 1. What changed in this version

### 1.1 Benchmark/statistical-test design

The statistical-test participant set now includes:

1. External benchmark models from `run_m5_full_benchmark.py`.
2. Proposed models from `run_m5_proposed_models.py`.
3. Two ablation models from `run_m5_ablation.py`:
   - `M0_AggHistOnly`
   - `M1_AggHistFutureSummary`

The full ablation run still also produces `M3_FullSkuTemporalCNN`, but the default statistical-test script filters ablation rows to include only `M0_AggHistOnly` and `M1_AggHistFutureSummary` as benchmark/statistical-comparison participants.

### 1.2 Bottom-up baseline replacement

The old bottom-up seasonal naive baseline has been replaced by:

```text
BottomUpGlobalHistGB
```

This model is trained globally at the child store-SKU level using gradient boosting, produces multi-horizon child-level forecasts, and then sums the child forecasts to the target aggregate level.

The old model name should no longer appear in final benchmark outputs:

```text
BottomUpSeasonalNaive
```

### 1.3 Cache version

The preprocessing cache now includes child-level future targets through the sample field:

```text
y_child_target
```

This is required by `BottomUpGlobalHistGB`. If old cache files are reused, rebuild the cache with `--force-rebuild-cache`.

---

## 2. Holdout and rolling consistency contract

The following split rules are used consistently.

### 2.1 Holdout mode

When `--mode holdout` is used:

| Role | Split name | Used by |
|---|---|---|
| Training | `final_train_samples` | benchmarks, proposed models, ablation models |
| Early stopping / internal validation | `internal_valid_samples` | proposed and ablation neural models |
| Final evaluation | `test_samples` | benchmarks, proposed models, ablation models |
| Fold id | `0` | all output rows |

Therefore, holdout results should only contain:

```text
fold = 0
```

### 2.2 Rolling mode

When `--mode rolling` is used:

| Role | Split name | Used by |
|---|---|---|
| Training | `rolling_fold_k_train_samples` | benchmarks, proposed models, ablation models |
| Validation/evaluation | `rolling_fold_k_valid_samples` | benchmarks, proposed models, ablation models |
| Fold id | `k = 1, 2, ...` | all output rows |

In rolling mode, each fold trains only on historical origins prior to that fold's validation origins. The validation fold is used as the rolling evaluation block.

### 2.3 Statistical-test alignment

The statistical test uses series-level rows when available. By default, the aligned statistical blocks are:

```text
task × agg_id
```

For rolling mode, per-origin/per-fold series metrics are averaged within each `task × agg_id × model` block before ranking models. This means the Friedman/Nemenyi tests compare models across aggregate series rather than across raw fold rows.

---

## 3. Important files

| File | Purpose |
|---|---|
| `scripts/prepare_m5_experiments.py` | M5 preprocessing, split generation, sample cache construction |
| `scripts/warm_m5_cache.py` | Pre-builds cache files before running experiments |
| `scripts/m5_benchmarks.py` | Aggregate-level, child-summary, and bottom-up benchmark models |
| `scripts/run_m5_full_benchmark.py` | Runs external benchmark models |
| `scripts/run_m5_proposed_models.py` | Runs proposed neural models |
| `scripts/run_m5_ablation.py` | Runs M0/M1/M3 ablation models and selected benchmark references |
| `scripts/tune_m5_proposed_models.py` | Optuna tuning for proposed models |
| `scripts/run_m5_stat_tests.py` | Friedman and Nemenyi statistical tests |
| `scripts/make_m5_paper_tables.py` | Builds paper-ready tables |
| `scripts/run_paper_experiment_plan.py` | Recommended end-to-end workflow runner |
| `scripts/check_experiment_consistency.py` | Post-run checker for holdout/rolling output consistency |

---

## 4. Required M5 data files

Set `DATA_DIR` to a folder containing:

```text
sales_train_evaluation.csv
calendar.csv
sell_prices.csv
```

If `sales_train_evaluation.csv` is unavailable, the code will try:

```text
sales_train_validation.csv
```

Recommended folder structure:

```text
project_root/
  scripts/
  data/m5/
    sales_train_evaluation.csv
    calendar.csv
    sell_prices.csv
  outputs/
  cache/
```

---

## 5. Cloud GPU execution through SSH

### 5.1 Connect to the cloud machine

From your local computer:

```bash
ssh user@YOUR_SERVER_IP
```

For first-time key setup, create a local SSH key if needed:

```bash
ssh-keygen -t ed25519 -C "m5-experiments"
```

Then add the public key to the cloud provider's SSH key panel or to `~/.ssh/authorized_keys` on the server.

### 5.2 Start a persistent terminal session

Use `tmux` so the experiment continues after your SSH session disconnects:

```bash
tmux new -s m5
```

Detach from `tmux`:

```bash
Ctrl+b d
```

Reconnect later:

```bash
tmux attach -t m5
```

### 5.3 Upload the project and data

From your local computer, use either `scp`:

```bash
scp scripts_modified_bottomup_ablation_stats_checked.zip user@YOUR_SERVER_IP:~/
```

or `rsync`:

```bash
rsync -avP scripts_modified_bottomup_ablation_stats_checked.zip user@YOUR_SERVER_IP:~/
```

On the cloud machine:

```bash
mkdir -p ~/m5_project
unzip ~/scripts_modified_bottomup_ablation_stats_checked.zip -d ~/m5_project
cd ~/m5_project
```

Upload the M5 data to, for example:

```text
~/m5_project/data/m5
```

### 5.4 Create the Python environment

Conda example:

```bash
conda create -n m5_forecast python=3.11 -y
conda activate m5_forecast
python -m pip install --upgrade pip
```

Install non-PyTorch dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib optuna wandb threadpoolctl tqdm
```

Install PyTorch with the command generated by the official PyTorch installation selector for your CUDA version. For many current Linux GPU environments, this has the following form:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
```

Replace `cuXXX` with the CUDA build recommended by the official selector.

### 5.5 Verify GPU availability

Check the NVIDIA driver and GPU visibility:

```bash
nvidia-smi
```

Check PyTorch CUDA availability:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

If `torch.cuda.is_available()` returns `False`, reinstall PyTorch using the correct CUDA wheel selected from the official PyTorch installation page.

---

## 6. Recommended environment variables

Before long runs, set:

```bash
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

These settings reduce CPU thread oversubscription when scikit-learn gradient boosting and PyTorch are used in the same workflow.

---

## 7. Define common paths

Run this on the cloud machine:

```bash
cd ~/m5_project
export DATA_DIR=~/m5_project/data/m5
export WORK_DIR=~/m5_project/outputs/m5_main
export CACHE_DIR=~/m5_project/cache/m5_cache
mkdir -p "$WORK_DIR" "$CACHE_DIR"
```

---

## 8. Fast sanity run

Start with a small sanity run before the full paper run. This checks data loading, model execution, GPU availability, and output writing.

```bash
python scripts/run_paper_experiment_plan.py \
  --data-dir "$DATA_DIR" \
  --work-dir "$WORK_DIR/sanity_debug" \
  --cache-dir "$CACHE_DIR" \
  --mode holdout \
  --phases cache,sanity \
  --tasks store_dept \
  --sanity-task store_dept \
  --store-sku-sample-frac 0.10 \
  --t-hist 56 \
  --quantiles "" \
  --quiet-progress \
  2>&1 | tee "$WORK_DIR/sanity_debug.log"
```

If this succeeds, proceed to the main run.

---

## 9. Recommended full holdout experiment

This is the clean final holdout experiment path.

```bash
python scripts/run_paper_experiment_plan.py \
  --data-dir "$DATA_DIR" \
  --work-dir "$WORK_DIR/holdout" \
  --cache-dir "$CACHE_DIR" \
  --mode holdout \
  --tasks store_dept,store_cat,state_dept \
  --proposed-models M3_FullSkuTemporalCNN,DeepSets,SetTransformer \
  --ablation-params-model M3_FullSkuTemporalCNN \
  --t-hist 56\
  --horizon 28 \
  --valid-size 4 \
  --test-size 4 \
  --internal-valid-size 3 \
  --store-sku-sample-frac 0.10 \
  --store-sku-sample-seed 42 \
  --stage1-n-trials 48 \
  --stage1-epochs 8 \
  --stage2-top-k 8 \
  --stage2-epochs 16 \
  --final-epochs 24 \
  --final-patience 6 \
  --stats-metric wspl \
  2>&1 | tee "$WORK_DIR/holdout_run.log"
```

Notes:

- `--store-sku-sample-frac 0.10` matches the computationally tractable 1/10 M5 store-SKU sample design.
- Use `--store-sku-sample-frac 1.0` only if the cloud machine has enough CPU memory and runtime budget.
- If you run without quantiles, set `--quantiles ""` and use `--stats-metric wrmsse`.

After completion, run:

```bash
python scripts/check_experiment_consistency.py \
  --work-dir "$WORK_DIR/holdout" \
  --mode holdout \
  --metric wspl
```

---

## 10. Recommended rolling experiment

For robustness or rolling-origin final evaluation:

```bash
python scripts/run_paper_experiment_plan.py \
  --data-dir "$DATA_DIR" \
  --work-dir "$WORK_DIR/rolling" \
  --cache-dir "$CACHE_DIR" \
  --mode rolling \
  --tasks store_dept,store_cat,state_dept \
  --proposed-models M3_FullSkuTemporalCNN,DeepSets,SetTransformer \
  --ablation-params-model M3_FullSkuTemporalCNN \
  --t-hist 56\
  --horizon 28 \
  --valid-size 4 \
  --test-size 4 \
  --internal-valid-size 3 \
  --store-sku-sample-frac 0.10 \
  --store-sku-sample-seed 42 \
  --stage1-n-trials 48 \
  --stage1-epochs 8 \
  --stage2-top-k 8 \
  --stage2-epochs 16 \
  --final-epochs 24 \
  --final-patience 6 \
  --stats-metric wspl \
  2>&1 | tee "$WORK_DIR/rolling_run.log"
```

Then check consistency:

```bash
python scripts/check_experiment_consistency.py \
  --work-dir "$WORK_DIR/rolling" \
  --mode rolling \
  --metric wspl
```

Important: the workflow intentionally tunes model hyperparameters using holdout-style internal validation, even when the final benchmark mode is rolling. The final benchmark, ablation, and statistical-test outputs follow the selected `--mode`.

---

## 11. Phase-by-phase execution

The end-to-end runner is easiest, but the phases can also be run separately.

### 11.1 Warm cache

```bash
python scripts/warm_m5_cache.py \
  --data-dir "$DATA_DIR" \
  --cache-dir "$CACHE_DIR" \
  --tasks store_dept,store_cat,state_dept \
  --holdout \
  --rolling \
  --t-hist 56\
  --horizon 28 \
  --valid-size 4 \
  --test-size 4 \
  --internal-valid-size 3 \
  --store-sku-sample-frac 0.10 \
  --store-sku-sample-seed 42
```

### 11.2 Tune proposed models

```bash
python scripts/tune_m5_proposed_models.py \
  --data-dir "$DATA_DIR" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$WORK_DIR/01_tuning/M3_FullSkuTemporalCNN" \
  --mode holdout \
  --tasks store_dept,store_cat,state_dept \
  --model-name M3_FullSkuTemporalCNN \
  --t-hist 56\
  --store-sku-sample-frac 0.10 \
  --stage1-n-trials 48 \
  --stage2-top-k 8
```

Repeat for:

```text
DeepSets
SetTransformer
```

### 11.3 Run ablation

```bash
python scripts/run_m5_ablation.py \
  --data-dir "$DATA_DIR" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$WORK_DIR/02_ablation" \
  --mode holdout \
  --tasks store_dept,store_cat,state_dept \
  --params-json "$WORK_DIR/01_tuning/M3_FullSkuTemporalCNN/best_params.json" \
  --t-hist 56\
  --store-sku-sample-frac 0.10
```

### 11.4 Run external benchmarks

```bash
python scripts/run_m5_full_benchmark.py \
  --data-dir "$DATA_DIR" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$WORK_DIR/03_benchmark_baselines" \
  --mode holdout \
  --tasks store_dept,store_cat,state_dept \
  --t-hist 56\
  --store-sku-sample-frac 0.10
```

### 11.5 Run proposed models

```bash
python scripts/run_m5_proposed_models.py \
  --data-dir "$DATA_DIR" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$WORK_DIR/04_benchmark_proposed/M3_FullSkuTemporalCNN" \
  --mode holdout \
  --tasks store_dept,store_cat,state_dept \
  --model-names M3_FullSkuTemporalCNN \
  --params-json "$WORK_DIR/01_tuning/M3_FullSkuTemporalCNN/best_params.json" \
  --t-hist 56\
  --store-sku-sample-frac 0.10
```

Repeat for `DeepSets` and `SetTransformer`, or use `run_paper_experiment_plan.py` to merge them automatically.

### 11.6 Run statistical tests

```bash
python scripts/run_m5_stat_tests.py \
  --benchmark-rows "$WORK_DIR/03_benchmark_baselines/all_holdout_rows.csv" \
  --benchmark-series-rows "$WORK_DIR/03_benchmark_baselines/all_holdout_series_rows.csv" \
  --proposed-rows "$WORK_DIR/04_benchmark_proposed/proposed_all_holdout_rows.csv" \
  --proposed-series-rows "$WORK_DIR/04_benchmark_proposed/proposed_all_holdout_series_rows.csv" \
  --ablation-rows "$WORK_DIR/02_ablation/all_ablation_rows_holdout.csv" \
  --ablation-series-rows "$WORK_DIR/02_ablation/all_ablation_series_rows_holdout.csv" \
  --ablation-models M0_AggHistOnly,M1_AggHistFutureSummary \
  --metric wspl \
  --analysis-level series \
  --series-block-cols task,agg_id \
  --output-dir "$WORK_DIR/05_stat_tests"
```

### 11.7 Build paper tables

```bash
python scripts/make_m5_paper_tables.py \
  --benchmark-rows "$WORK_DIR/03_benchmark_baselines/all_holdout_rows.csv" \
  --proposed-rows "$WORK_DIR/04_benchmark_proposed/proposed_all_holdout_rows.csv" \
  --ablation-rows "$WORK_DIR/02_ablation/all_ablation_rows_holdout.csv" \
  --output-dir "$WORK_DIR/06_paper_tables"
```

---

## 12. Monitoring long runs

In another SSH session:

```bash
watch -n 5 nvidia-smi
```

Check the log:

```bash
tail -f "$WORK_DIR/holdout_run.log"
```

Check disk usage:

```bash
du -sh "$CACHE_DIR" "$WORK_DIR"
```

---

## 13. Expected output files

For `holdout` mode, key files are:

```text
$WORK_DIR/holdout/02_ablation/all_ablation_rows_holdout.csv
$WORK_DIR/holdout/02_ablation/all_ablation_series_rows_holdout.csv
$WORK_DIR/holdout/03_benchmark_baselines/all_holdout_rows.csv
$WORK_DIR/holdout/03_benchmark_baselines/all_holdout_series_rows.csv
$WORK_DIR/holdout/04_benchmark_proposed/proposed_all_holdout_rows.csv
$WORK_DIR/holdout/04_benchmark_proposed/proposed_all_holdout_series_rows.csv
$WORK_DIR/holdout/05_stat_tests/Table_stats_mean_spl_series.csv
$WORK_DIR/holdout/05_stat_tests/Figure_cd_diagram_mean_spl_series.png
$WORK_DIR/holdout/06_paper_tables/Table2_main_point.tex
$WORK_DIR/holdout/06_paper_tables/Table3_main_quantile.tex
$WORK_DIR/holdout/06_paper_tables/Table4_ablation.tex
```

For `rolling` mode, replace `holdout` with `rolling` in file names.

---

## 14. Troubleshooting

### 14.1 `KeyError: y_child_target`

Cause: old cache files were created before the bottom-up global store-SKU model was added.

Fix:

```bash
python scripts/warm_m5_cache.py \
  --data-dir "$DATA_DIR" \
  --cache-dir "$CACHE_DIR" \
  --tasks store_dept,store_cat,state_dept \
  --holdout \
  --rolling \
  --force-rebuild-cache
```

### 14.2 CUDA is not used

Check:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is unavailable, reinstall PyTorch using the official CUDA-specific wheel command.

### 14.3 Out-of-memory during preprocessing

Use the sampled store-SKU setting:

```bash
--store-sku-sample-frac 0.10
```

Also use a shared cache directory and avoid `--force-rebuild-cache` unless the preprocessing logic changes.

### 14.4 Statistical test fails because no complete blocks remain

Check that all models were run with the same settings:

```text
--tasks
--mode
--t-hist
--horizon
--valid-size
--test-size
--internal-valid-size
--store-sku-sample-frac
--store-sku-sample-seed
```

Then run:

```bash
python scripts/check_experiment_consistency.py \
  --work-dir "$WORK_DIR/holdout" \
  --mode holdout \
  --metric wspl
```

### 14.5 Quantile outputs missing

If you used:

```bash
--quantiles ""
```

then WSPL outputs are intentionally absent. Use:

```bash
--stats-metric wrmsse
```

for statistical tests.

---

## 15. Recommended paper interpretation

Use `holdout` as the primary final test if the paper emphasizes strict held-out evaluation. Use `rolling` as a robustness check because it evaluates performance stability across multiple rolling-origin blocks.

For statistical comparisons, report series-level Friedman/Nemenyi results based on `task × agg_id` blocks. This aligns with the paper's goal of comparing relative model performance across aggregate series rather than treating every horizon/origin row as independent.


---

## 12. Large-sample tuning for 30%, 50%, and 100% SKU samples

The tuner `scripts/tune_m5_proposed_models.py` now supports memory-aware search profiles:

| SKU sample fraction | Auto profile | Main purpose |
|---:|---|---|
| `0.10` | `sample10` | Initial tuning / sanity tuning |
| `0.30` | `large30` | Main large-sample retuning |
| `0.50` | `large50` | Stability confirmation |
| `1.00` | `full100` | Final narrow confirmation |

Use:

```bash
--search-profile auto
```

or specify the profile manually:

```bash
--search-profile large30
--search-profile large50
--search-profile full100
```

The package also includes the 10% tuned parameter file:

```text
best_params_all_models.json
```

This file can be used as a warm start. The tuner accepts this all-models JSON and automatically extracts the parameters for the selected `--model-name`.

### 12.1 Recommended 30% tuning commands

Use 30% as the main large-sample retuning stage.

```bash
for MODEL in M3_FullSkuTemporalCNN DeepSets SetTransformer; do
  python scripts/tune_m5_proposed_models.py \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR/01_tuning_30pct/${MODEL}" \
    --mode holdout \
    --tasks store_dept,store_cat,state_dept \
    --model-name "$MODEL" \
    --search-profile auto \
    --warm-start-params best_params_all_models.json \
    --t-hist 56\
    --horizon 28 \
    --valid-size 4 \
    --test-size 4 \
    --internal-valid-size 3 \
    --store-sku-sample-frac 0.30 \
    --store-sku-sample-seed 42 \
    --stage1-n-trials 40 \
    --stage1-train-frac 0.40 \
    --stage1-epochs 10 \
    --stage1-patience 4 \
    --stage2-top-k 8 \
    --stage2-train-frac 1.00 \
    --stage2-epochs 20 \
    --stage2-patience 6 \
    --final-epochs 32 \
    --final-patience 8 \
    --storage "sqlite:///$WORK_DIR/optuna_${MODEL}_30pct.db" \
    2>&1 | tee "$WORK_DIR/tune_${MODEL}_30pct.log"
done
```

### 12.2 Recommended 50% confirmation commands

Use fewer trials because the 30% search should already identify the stable region.

```bash
for MODEL in M3_FullSkuTemporalCNN DeepSets SetTransformer; do
  python scripts/tune_m5_proposed_models.py \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR/01_tuning_50pct/${MODEL}" \
    --mode holdout \
    --tasks store_dept,store_cat,state_dept \
    --model-name "$MODEL" \
    --search-profile auto \
    --warm-start-params "$WORK_DIR/01_tuning_30pct/${MODEL}/best_params.json" \
    --t-hist 56\
    --store-sku-sample-frac 0.50 \
    --store-sku-sample-seed 42 \
    --stage1-n-trials 20 \
    --stage1-train-frac 0.40 \
    --stage1-epochs 10 \
    --stage1-patience 4 \
    --stage2-top-k 6 \
    --stage2-train-frac 1.00 \
    --stage2-epochs 20 \
    --stage2-patience 6 \
    --final-epochs 32 \
    --final-patience 8 \
    --storage "sqlite:///$WORK_DIR/optuna_${MODEL}_50pct.db" \
    2>&1 | tee "$WORK_DIR/tune_${MODEL}_50pct.log"
done
```

### 12.3 Recommended 100% final confirmation commands

For the full SKU sample, do not run a broad architecture search. Use the `full100` profile and a small number of trials.

```bash
for MODEL in M3_FullSkuTemporalCNN DeepSets SetTransformer; do
  python scripts/tune_m5_proposed_models.py \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR/01_tuning_100pct/${MODEL}" \
    --mode holdout \
    --tasks store_dept,store_cat,state_dept \
    --model-name "$MODEL" \
    --search-profile full100 \
    --warm-start-params "$WORK_DIR/01_tuning_50pct/${MODEL}/best_params.json" \
    --t-hist 56\
    --store-sku-sample-frac 1.00 \
    --store-sku-sample-seed 42 \
    --stage1-n-trials 12 \
    --stage1-train-frac 0.30 \
    --stage1-epochs 8 \
    --stage1-patience 3 \
    --stage2-top-k 4 \
    --stage2-train-frac 0.80 \
    --stage2-epochs 16 \
    --stage2-patience 5 \
    --final-epochs 32 \
    --final-patience 8 \
    --storage "sqlite:///$WORK_DIR/optuna_${MODEL}_100pct.db" \
    2>&1 | tee "$WORK_DIR/tune_${MODEL}_100pct.log"
done
```

### 12.4 What the large-sample tuner now does

The revised tuner:

- automatically maps `--store-sku-sample-frac` to an appropriate search profile;
- narrows learning rate, dropout, weight decay, hidden dimension, CNN blocks, and batch-size ranges as the sample fraction increases;
- uses smaller, safer batch-size choices for Set Transformer at 50% and 100%;
- can load the 10% all-models parameter JSON and enqueue the corresponding model's parameters as the first Optuna trial;
- records the active search space in `tuning_run_config.json`;
- stores Set Transformer parameters such as `num_heads`, `num_sab_layers`, and `num_inducing_points` in `best_params.json`;
- prunes out-of-memory Optuna trials during stage 1;
- retries stage-2 and final confirmation with smaller batch sizes if a promising configuration runs out of GPU memory.

For final paper reporting, use the 30% best parameters as the main large-sample retuning result, and use 50%/100% as stability checks unless you have enough cloud budget for full retuning.
