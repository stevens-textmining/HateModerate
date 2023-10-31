#!/bin/bash

MODEL_NAME="roberta-base"
MODEL_TYPE="roberta"

LEARNING_RATES=("1e-6" "2e-6" "3e-6" "5e-6" "8e-6" "1e-5")
EPOCHS=("1" "2" "3")
INCLUDE_OPTIONS=("True" "False")

for LR in "${LEARNING_RATES[@]}"; do
  for EPOCH in "${EPOCHS[@]}"; do
    for INCLUDE_HATEMODERATE in "${INCLUDE_OPTIONS[@]}"; do
      echo "Training with LR = $LR, EPOCH = $EPOCH, INCLUDE_HATEMODERATE = $INCLUDE_HATEMODERATE"
      python fine_tune/training/fine_tune.py "$MODEL_NAME" "$LR" "$EPOCH" "$MODEL_TYPE" "$INCLUDE_HATEMODERATE"
    done
  done
done
