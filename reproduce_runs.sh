#!/usr/bin/env bash
# Reproduce the runs behind the challenge submission.
#
# 1. the two cross-center transfer evaluations
# 2. the pooled held-out evaluation, over the five ensemble seeds
# 3. the five final models trained on all labeled data, exported for the container
set -e

SEEDS=(42 43 44 45 46)
POOLING="${POOLING:-attention}"
WEIGHTS="${BACKBONE_WEIGHTS:-resources/gastronet_dinov2_vitb.pth}"
COMMON=(--data_root data/train --dinov2_repo third_party/dinov2 --backbone_weights "$WEIGHTS" --pooling "$POOLING")

echo "== cross-center transfer =="
for CSV in center1_train_center2_test center2_train_center1_test; do
  python -m training.run_training --split cross_center --split_csv "data/splits/${CSV}.csv" --seed 42 "${COMMON[@]}"
done

echo "== pooled held-out evaluation =="
for SEED in "${SEEDS[@]}"; do
  python -m training.run_training --split pooled --split_csv data/splits/pooled_holdout.csv --seed "$SEED" "${COMMON[@]}"
done
python -m validation.run_validation results/predictions_pooled_${POOLING}_seed*.csv

echo "== final models for the container =="
mkdir -p inference/resources
for SEED in "${SEEDS[@]}"; do
  python -m training.run_training --split final --split_csv data/splits/pooled_holdout.csv --seed "$SEED" \
    --export "inference/resources/dinov2_lora_attnmil_neoplasia_seed${SEED}.pt" "${COMMON[@]}"
done

echo "done"
