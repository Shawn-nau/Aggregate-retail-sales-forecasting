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
CACHE_DIR=/root/autodl-tmp/m5/cache/50pct
WORK_DIR=/root/autodl-tmp/m5/work/50pct
TUNED_30PCT=/root/autodl-tmp/m5/work/30pct/01_tuning
FRAC=0.50
SEED=42

mkdir -p "$CACHE_DIR" "$WORK_DIR"

echo "=== Phase 1: Cache (50% SKU) ==="
$PYTHON $SCRIPTS/warm_m5_cache.py \
  --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
  --tasks store_dept,store_cat,state_dept --holdout \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac $FRAC --store-sku-sample-seed $SEED

for MODEL in M3_FullSkuTemporalCNN DeepSets SetTransformer; do
  echo "=== Phase 2: Tuning $MODEL (50%) ==="
  $PYTHON $SCRIPTS/tune_m5_proposed_models.py \
    --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR/01_tuning/$MODEL" \
    --mode holdout --tasks store_dept,store_cat,state_dept \
    --model-name "$MODEL" --search-profile auto \
    --warm-start-params "$TUNED_30PCT/$MODEL/best_params.json" \
    --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
    --store-sku-sample-frac $FRAC --store-sku-sample-seed $SEED \
    --stage1-n-trials 20 --stage1-train-frac 0.40 --stage1-epochs 10 --stage1-patience 4 \
    --stage2-top-k 6 --stage2-train-frac 1.00 --stage2-epochs 20 --stage2-patience 6 \
    --final-epochs 32 --final-patience 8 \
    --storage "sqlite:///$WORK_DIR/optuna_$MODEL.db"
done

echo "=== Phase 3: Ablation + Benchmark + Proposed + Stats + Tables ==="
$PYTHON $SCRIPTS/run_paper_experiment_plan.py \
  --data-dir "$DATA_DIR" --work-dir "$WORK_DIR" --cache-dir "$CACHE_DIR" \
  --mode holdout --tasks store_dept,store_cat,state_dept \
  --proposed-models M3_FullSkuTemporalCNN,DeepSets,SetTransformer \
  --ablation-params-model M3_FullSkuTemporalCNN \
  --phases ablation,benchmark,stats,tables \
  --t-hist 56 --horizon 28 --valid-size 4 --test-size 4 --internal-valid-size 3 \
  --store-sku-sample-frac $FRAC --store-sku-sample-seed $SEED \
  --final-epochs 24 --final-patience 6 --stats-metric wspl

echo "=== Phase 4: Consistency ==="
$PYTHON $SCRIPTS/check_experiment_consistency.py \
  --work-dir "$WORK_DIR" --mode holdout --metric wspl

echo "=== ALL DONE ==="
