#!/bin/bash

datasets=("../Hate_Check.csv"
          "../../postprocess/cardiffnlp_test_sets/test_hatE.csv"
          "../../postprocess/cardiffnlp_test_sets/test_htpo.csv"
          "../../postprocess/cardiffnlp_test_sets/test_hatex.csv")

models=("../card_only_roberta-base_lr=5e-06_epoch=2"
        "../roberta-base_lr=5e-06_epoch=2")

labels=("LABEL_1") # HATE

echo "dataset,model,label,metric" > results.csv


for dataset in "${datasets[@]}"; do
    for ((i=0; i<${#models[@]}; i++)); do
        export CUDA_VISIBLE_DEVICES=$((i % $(nvidia-smi -L | wc -l)))
        metrics=$(python -c "import os, torch; from utils import predict_hate_label, calculate_matching, process_dataset; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); process_dataset('$dataset', '${models[$i]}', '$labels', device)")
        echo "$dataset,${models[$i]}, $labels, $metrics" >> results.csv
    done
done

