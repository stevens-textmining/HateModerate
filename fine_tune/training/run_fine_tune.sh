#!/bin/bash

MODEL_NAME="roberta-base"
MODEL_TYPE="roberta"

LEARNING_RATES=("1e-6" "2e-6" "3e-6" "5e-6" "8e-6" "1e-5")
EPOCHS=("1" "2" "3")
INCLUDE_OPTIONS=("True" "False")

for LR in "${LEARNING_RATES[@]}"; do
  for EPOCH in "${EPOCHS[@]}"; do
    echo "Training with LR = $LR, EPOCHS = $EPOCH, INCLUDE = False"
    python fine_tune/training/fine_tune.py --model_name "$MODEL_NAME" --learning_rate "$LR" --n_epoch "$EPOCH" --model_type "$MODEL_TYPE" --no-include

    echo "Training with LR = $LR, EPOCHS = $EPOCH, INCLUDE = True"
    python fine_tune/training/fine_tune.py --model_name "$MODEL_NAME" --learning_rate "$LR" --n_epoch "$EPOCH" --model_type "$MODEL_TYPE" --include
  done
done
