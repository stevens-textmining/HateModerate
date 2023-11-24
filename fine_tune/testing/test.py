import torch
from utils import process_dataset
import contextlib
import io


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets=("fine_tune/datasets/testing/Hate_Check.csv",
          "fine_tune/datasets/testing/hatemoderate_test.csv",
          "fine_tune/datasets/testing/test_hatE.csv",
          "fine_tune/datasets/testing/test_hatex.csv",
          "fine_tune/datasets/testing/test_htpo.csv")


labels = "LABEL_1"

models=(
        "roberta-base_lr=1e-05_epoch=3_include_hatemoderate",
        "roberta-base_lr=1e-05_epoch=3_include_hatemoderate/checkpoint-7652-epoch-1",
        "roberta-base_lr=1e-05_epoch=3_include_hatemoderate/checkpoint-15304-epoch-2",

        "roberta-base_lr=1e-05_epoch=3_no-include_hatemoderate",
        "roberta-base_lr=1e-05_epoch=3_no-include_hatemoderate/checkpoint-7581-epoch-1",
        "roberta-base_lr=1e-05_epoch=3_no-include_hatemoderate/checkpoint-15162-epoch-2",
        "roberta-base_lr=2e-05_epoch=4_include_hatemoderate",
        "roberta-base_lr=2e-05_epoch=4_include_hatemoderate/checkpoint-7652-epoch-1",
        "roberta-base_lr=2e-05_epoch=4_include_hatemoderate/checkpoint-15304-epoch-2",
        "roberta-base_lr=2e-05_epoch=4_include_hatemoderate/checkpoint-22956-epoch-3",

        "roberta-base_lr=2e-05_epoch=4_no-include_hatemoderate",
        "roberta-base_lr=2e-05_epoch=4_no-include_hatemoderate/checkpoint-7581-epoch-1",
        "roberta-base_lr=2e-05_epoch=4_no-include_hatemoderate/checkpoint-15162-epoch-2",
        "roberta-base_lr=2e-05_epoch=4_no-include_hatemoderate/checkpoint-22743-epoch-3",

        "roberta-base_lr=1e-06_epoch=3_include_hatemoderate",
        "roberta-base_lr=1e-06_epoch=3_include_hatemoderate/checkpoint-7652-epoch-1",
        "roberta-base_lr=1e-06_epoch=3_include_hatemoderate/checkpoint-15304-epoch-2",

        "roberta-base_lr=1e-06_epoch=3_no-include_hatemoderate",
        "roberta-base_lr=1e-06_epoch=3_no-include_hatemoderate/checkpoint-7581-epoch-1",
        "roberta-base_lr=1e-06_epoch=3_no-include_hatemoderate/checkpoint-15162-epoch-2",

        "roberta-base_lr=2e-06_epoch=3_include_hatemoderate",
        "roberta-base_lr=2e-06_epoch=3_include_hatemoderate/checkpoint-7652-epoch-1",
        "roberta-base_lr=2e-06_epoch=3_include_hatemoderate/checkpoint-15304-epoch-2",

        "roberta-base_lr=2e-06_epoch=3_no-include_hatemoderate",
        "roberta-base_lr=2e-06_epoch=3_no-include_hatemoderate/checkpoint-7581-epoch-1",
        "roberta-base_lr=2e-06_epoch=3_no-include_hatemoderate/checkpoint-15162-epoch-2",

        "roberta-base_lr=3e-06_epoch=3_include_hatemoderate",
        "roberta-base_lr=3e-06_epoch=3_include_hatemoderate/checkpoint-7652-epoch-1",
        "roberta-base_lr=3e-06_epoch=3_include_hatemoderate/checkpoint-15304-epoch-2",

        "roberta-base_lr=3e-06_epoch=3_no-include_hatemoderate",
        "roberta-base_lr=3e-06_epoch=3_no-include_hatemoderate/checkpoint-7581-epoch-1",
        "roberta-base_lr=3e-06_epoch=3_no-include_hatemoderate/checkpoint-15162-epoch-2",

        "roberta-base_lr=5e-06_epoch=3_include_hatemoderate",
        "roberta-base_lr=5e-06_epoch=3_include_hatemoderate/checkpoint-7652-epoch-1",
        "roberta-base_lr=5e-06_epoch=3_include_hatemoderate/checkpoint-15304-epoch-2",

        "roberta-base_lr=5e-06_epoch=3_no-include_hatemoderate",
        "roberta-base_lr=5e-06_epoch=3_no-include_hatemoderate/checkpoint-7581-epoch-1",
        "roberta-base_lr=5e-06_epoch=3_no-include_hatemoderate/checkpoint-15162-epoch-2",
)




output_file = "test_results.txt"

with open(output_file, "a") as f:
    for dataset in datasets:
        for model in models:
            f.write(f"Dataset: {dataset}, Model: {model}\n")

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                process_dataset(dataset, model, labels, device)

            output = buffer.getvalue()
            f.write(output)
            f.write("\n")