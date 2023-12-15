import torch
from utils import process_dataset
import contextlib
import io


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets=(
    "fine_tune/datasets/testing/Hate_Check.csv",
      "fine_tune/datasets/testing/hatemoderate_test.csv",
      "fine_tune/datasets/testing/test_hatE.csv",
      "fine_tune/datasets/testing/test_hatex.csv",
      "fine_tune/datasets/testing/test_htpo.csv"
      )


labels = "LABEL_1"

models=(

    "roberta-base_lr=1e-05_epoch=4_batch_size=32_include_hatemoderate",

    "roberta-base_lr=1e-05_epoch=4_batch_size=32_rebalance",

    "roberta-base_lr=2e-05_epoch=4_batch_size=32_include_hatemoderate",

    "roberta-base_lr=2e-05_epoch=4_batch_size=32_rebalance",

    "roberta-base_lr=2e-05_epoch=4_batch_size=32_no-include_hatemoderate",


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