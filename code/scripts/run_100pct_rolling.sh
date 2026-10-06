#!/bin/bash
set -e
cd /root/autodl-tmp/m5
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

PYTHON=/root/miniconda3/bin/python
SCRIPTS=/root/autodl-tmp/m5/scripts
DATA_DIR=/root/autodl-tmp/m5/data
CACHE_DIR=/root/autodl-tmp/m5/cache
WORK_DIR=/root/autodl-tmp/m5/work/100pct_rolling
TUNED_100PCT=/root/autodl-tmp/m5/work/100pct/01_tuning
FRAC=1.00
SEED=42
MODE=rolling

mkdir -p "$CACHE_DIR" "$WORK_DIR"

echo "=== Phase 1: Warm cache (100% SKU, holdout + rolling) ==="
$PYTHON $SCRIPTS/warm_m5_cache.py \
  --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
  --tasks store_dept,store_cat,state_dept --holdout --rolling \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac $FRAC --store-sku-sample-seed $SEED

echo "=== Phase 2: Ablation + Benchmark + Proposed + Stats + Tables (rolling) ==="
$PYTHON $SCRIPTS/run_paper_experiment_plan.py \
  --data-dir "$DATA_DIR" --work-dir "$WORK_DIR" --cache-dir "$CACHE_DIR" \
  --mode "$MODE" --tasks store_dept,store_cat,state_dept \
  --proposed-models M3_FullSkuTemporalCNN,DeepSets,SetTransformer \
  --ablation-params-model M3_FullSkuTemporalCNN \
  --phases ablation,benchmark,stats,tables \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac $FRAC --store-sku-sample-seed $SEED \
  --final-epochs 24 --final-patience 6 --stats-metric wspl

echo "=== Phase 3: Consistency check ==="
$PYTHON $SCRIPTS/check_experiment_consistency.py \
  --work-dir "$WORK_DIR" --mode "$MODE" --metric wspl

echo "=== ALL DONE ==="
