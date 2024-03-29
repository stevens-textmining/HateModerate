#!/bin/bash

MODEL_NAME="roberta-base"
MODEL_TYPE="roberta"

#LEARNING_RATES=("1e-6" "2e-6" "3e-6" "5e-6" "1e-5" "2e-5")
EPOCHS=("2" "3" "4")
LEARNING_RATES=("2e-5" "1e-5")
#EPOCHS=("2")
BATCH_SIZES=("4" "16" "32")
#BATCH_SIZES=("32")
INCLUDE_OPTIONS=("True" "False")



for LR in "${LEARNING_RATES[@]}"; do
  for EPOCH in "${EPOCHS[@]}"; do
    for INCLUDE in "${INCLUDE_OPTIONS[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        if [ "$INCLUDE" == "True" ]; then
          echo "Training with LR = $LR, EPOCHS = $EPOCH, BATCH_SIZE = $BATCH_SIZE, INCLUDE = True"
          python fine_tune/training/fine_tune.py --model_name "$MODEL_NAME" --learning_rate "$LR" --n_epoch "$EPOCH" --batch_size "$BATCH_SIZE" --model_type "$MODEL_TYPE" --include
        else
          echo "Training with LR = $LR, EPOCHS = $EPOCH, BATCH_SIZE = $BATCH_SIZE, INCLUDE = False"
          python fine_tune/training/fine_tune.py --model_name "$MODEL_NAME" --learning_rate "$LR" --n_epoch "$EPOCH" --batch_size "$BATCH_SIZE" --model_type "$MODEL_TYPE" --no-include
        fi
      done
    done
  done
done