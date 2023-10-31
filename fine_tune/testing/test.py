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
models=("card_only_roberta-base_lr=5e-06_epoch=2",
        "roberta-base_lr=5e-06_epoch=2")


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