#!/bin/bash
# =============================================================================
# run_robustness_5seeds.sh — 5-repeat robustness experiment (10% SKU, 5 seeds)
#
# NOTE: The paths below are from the original AutoDL cloud execution
# environment (/root/autodl-tmp/m5). To run on your own machine:
#   1. Replace /root/autodl-tmp/m5 with your repo root path
#   2. Replace /root/miniconda3/bin/python with your Python interpreter
#   3. Comment out or adjust the conda activate lines for your setup
#
# The equivalent Python commands are documented in README.md — those serve as
# the authoritative reproduction instructions.
# =============================================================================
set -e
cd /root/autodl-tmp/m5
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PYTHON=/root/miniconda3/bin/python
SCRIPTS=/root/autodl-tmp/m5/scripts
DATA_DIR=/root/autodl-tmp/m5/data
TUNED_PARAMS_DIR=/root/autodl-tmp/m5/work/holdout/01_tuning
BASE_WORK=/root/autodl-tmp/m5/work/holdout/robustness_5seeds
CACHES_DIR=/root/autodl-tmp/m5/cache/robustness_5seeds

TASKS=store_dept,store_cat,state_dept
MODE=holdout
QUANTILES="0.005,0.025,0.165,0.25,0.5,0.75,0.835,0.975,0.995"
ABLATION_PARAMS_MODEL=M3_FullSkuTemporalCNN
ABLATION_PARAMS_JSON="$TUNED_PARAMS_DIR/$ABLATION_PARAMS_MODEL/best_params.json"

SEEDS=(142 242 342 442 542)

echo "=============================================="
echo "ROBUSTNESS: 5-REPEAT HOLDOUT WITH VARYING SEEDS"
echo "=============================================="
echo "Data-sampling seeds: ${SEEDS[*]}"
echo "Tuned params source: $TUNED_PARAMS_DIR"
echo "Base work dir:       $BASE_WORK"
echo ""

for SEED in "${SEEDS[@]}"; do
    CACHE_DIR="$CACHES_DIR/seed_$SEED"
    WORK_DIR="$BASE_WORK/seed_$SEED"
    mkdir -p "$CACHE_DIR" "$WORK_DIR"

    echo "############################################################"
    echo "### REPEAT: SEED=$SEED"
    echo "### CACHE: $CACHE_DIR"
    echo "### WORK:  $WORK_DIR"
    echo "############################################################"

    # --- Phase 1: Build cache with this seed ---
    echo "[$(date)] Phase 1/6: Building cache (seed=$SEED)..."
    $PYTHON $SCRIPTS/warm_m5_cache.py \
        --data-dir "$DATA_DIR" \
        --tasks "$TASKS" \
        --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
        --gap 0 --min-train-origins 20 --min-final-train-origins 20 \
        --sku-universe-mode history \
        --store-sku-sample-frac 0.10 --store-sku-sample-seed "$SEED" \
        --holdout \
        --cache-dir "$CACHE_DIR"

    # --- Phase 2: Ablation ---
    echo "[$(date)] Phase 2/6: Ablation (seed=$SEED)..."
    $PYTHON $SCRIPTS/run_m5_ablation.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$WORK_DIR/02_ablation" \
        --mode "$MODE" --tasks "$TASKS" \
        --recent-window 14 --lag-windows 7,14,28,56 --stat-windows 7,28,56 \
        --params-json "$ABLATION_PARAMS_JSON" \
        --quantiles "$QUANTILES" --epochs 24 --patience 6 \
        --seed "$SEED" \
        --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
        --gap 0 --min-train-origins 20 --min-final-train-origins 20 \
        --sku-universe-mode history \
        --store-sku-sample-frac 0.10 --store-sku-sample-seed "$SEED" \
        --cache-dir "$CACHE_DIR"

    # --- Phase 3: Benchmark baselines ---
    echo "[$(date)] Phase 3/6: Benchmark baselines (seed=$SEED)..."
    $PYTHON $SCRIPTS/run_m5_full_benchmark.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$WORK_DIR/03_benchmark_baselines" \
        --mode "$MODE" --tasks "$TASKS" \
        --quantiles "$QUANTILES" --random-state "$SEED" \
        --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
        --gap 0 --min-train-origins 20 --min-final-train-origins 20 \
        --sku-universe-mode history \
        --store-sku-sample-frac 0.10 --store-sku-sample-seed "$SEED" \
        --cache-dir "$CACHE_DIR"

    # --- Phase 4: Proposed models (using original tuned params) ---
    echo "[$(date)] Phase 4/6: Proposed models (seed=$SEED)..."
    for MODEL in M3_FullSkuTemporalCNN DeepSets SetTransformer; do
        PARAMS_JSON="$TUNED_PARAMS_DIR/$MODEL/best_params.json"
        if [ ! -f "$PARAMS_JSON" ]; then
            echo "WARNING: $PARAMS_JSON not found, skipping $MODEL"
            continue
        fi
        echo "  -> Model: $MODEL"
        $PYTHON $SCRIPTS/run_m5_proposed_models.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$WORK_DIR/04_benchmark_proposed/$MODEL" \
            --mode "$MODE" --tasks "$TASKS" \
            --quantiles "$QUANTILES" --model-names "$MODEL" \
            --recent-window 14 --lag-windows 7,14,28,56 --stat-windows 7,28,56 \
            --params-json "$PARAMS_JSON" \
            --seed "$SEED" \
            --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
            --gap 0 --min-train-origins 20 --min-final-train-origins 20 \
            --sku-universe-mode history \
            --store-sku-sample-frac 0.10 --store-sku-sample-seed "$SEED" \
            --cache-dir "$CACHE_DIR"
    done

    # --- Merge proposed model CSVs (since we ran models separately) ---
    echo "[$(date)] Merging proposed model CSVs..."
    $PYTHON -c "
import pandas as pd
from pathlib import Path
base = Path('$WORK_DIR/04_benchmark_proposed')
mode = '$MODE'
rows = []; series_rows = []; point_rows = []; quantile_rows = []
for model_dir in sorted(base.iterdir()):
    if not model_dir.is_dir(): continue
    r = model_dir / f'proposed_all_{mode}_rows.csv'
    if r.exists(): rows.append(pd.read_csv(r))
    s = model_dir / f'proposed_all_{mode}_series_rows.csv'
    if s.exists(): series_rows.append(pd.read_csv(s))
    p = model_dir / f'proposed_point_summary_{mode}.csv'
    if p.exists(): point_rows.append(pd.read_csv(p))
    q = model_dir / f'proposed_quantile_summary_{mode}.csv'
    if q.exists(): quantile_rows.append(pd.read_csv(q))
if rows:
    pd.concat(rows, ignore_index=True).to_csv(base / f'proposed_all_{mode}_rows.csv', index=False)
if series_rows:
    pd.concat(series_rows, ignore_index=True).to_csv(base / f'proposed_all_{mode}_series_rows.csv', index=False)
if point_rows:
    pd.concat(point_rows, ignore_index=True).to_csv(base / f'proposed_point_summary_{mode}.csv', index=False)
if quantile_rows:
    pd.concat(quantile_rows, ignore_index=True).to_csv(base / f'proposed_quantile_summary_{mode}.csv', index=False)
print(f'Merged {len(rows)} model dirs into {base}')
"

    # --- Phase 5: Statistical tests ---
    echo "[$(date)] Phase 5/6: Stats (seed=$SEED)..."
    $PYTHON $SCRIPTS/run_m5_stat_tests.py \
        --benchmark-rows "$WORK_DIR/03_benchmark_baselines/all_${MODE}_rows.csv" \
        --proposed-rows "$WORK_DIR/04_benchmark_proposed/proposed_all_${MODE}_rows.csv" \
        --ablation-rows "$WORK_DIR/02_ablation/all_ablation_rows_${MODE}.csv" \
        --benchmark-series-rows "$WORK_DIR/03_benchmark_baselines/all_${MODE}_series_rows.csv" \
        --proposed-series-rows "$WORK_DIR/04_benchmark_proposed/proposed_all_${MODE}_series_rows.csv" \
        --ablation-series-rows "$WORK_DIR/02_ablation/all_ablation_series_rows_${MODE}.csv" \
        --ablation-models M0_AggHistOnly,M1_AggHistFutureSummary \
        --metric wspl --analysis-level series --series-block-cols task,agg_id \
        --output-dir "$WORK_DIR/05_stat_tests"

    # --- Phase 6: Paper tables ---
    echo "[$(date)] Phase 6/6: Tables (seed=$SEED)..."
    $PYTHON $SCRIPTS/make_m5_paper_tables.py \
        --benchmark-rows "$WORK_DIR/03_benchmark_baselines/all_${MODE}_rows.csv" \
        --proposed-rows "$WORK_DIR/04_benchmark_proposed/proposed_all_${MODE}_rows.csv" \
        --ablation-rows "$WORK_DIR/02_ablation/all_ablation_rows_${MODE}.csv" \
        --output-dir "$WORK_DIR/06_paper_tables"

    echo "[$(date)] SEED=$SEED COMPLETE"
    echo ""
done

echo "=============================================="
echo "ALL 5 ROBUSTNESS REPEATS COMPLETE"
echo "=============================================="
