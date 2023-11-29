import torch
from utils import process_dataset
import contextlib
import io


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets=(
    "fine_tune/datasets/testing/Hate_Check.csv",
          "fine_tune/datasets/testing/hatemoderate_test.csv",
          # "fine_tune/datasets/testing/test_hatE.csv",
          # "fine_tune/datasets/testing/test_hatex.csv",
          # "fine_tune/datasets/testing/test_htpo.csv"
          )


labels = "LABEL_1"

models=(
        # "roberta-base_lr=1e-05_epoch=4_batch_size=4_include_hatemoderate/checkpoint-61212-epoch-1",
        # "roberta-base_lr=1e-05_epoch=4_batch_size=4_include_hatemoderate/checkpoint-122424-epoch-2",
        # "roberta-base_lr=1e-05_epoch=4_batch_size=4_include_hatemoderate/checkpoint-183636-epoch-3",
        # "roberta-base_lr=1e-05_epoch=4_batch_size=4_include_hatemoderate",
        "best_hp_card+dynahate",
        "best_hp_card+dynahate+nonDH-HM",
        # "best_hp_all_train_gpt",


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