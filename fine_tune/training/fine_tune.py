import sys
import os
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd


def train_hate_model(model_name,
                     learning_rate=1e-6,
                     n_epoch=3,
                     train_file="fine_tune/datasets/training/hatemoderate_train.csv",
                     test_file="fine_tune/datasets/testing/hatemoderate_test.csv",
                     compare_datasets="fine_tune/datasets/training/cardiffnlp.pkl",
                     include = True,
                     model_type = "roberta"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_args = ClassificationArgs()

    model_args.learning_rate = learning_rate
    model_args.num_train_epochs = n_epoch
    model_args.train_batch_size = 32
    model_args.eval_batch_size = 32
    model_args.n_gpu = 4
    model_args.output_dir = "{}_lr={}_epoch={}_hatemoderate".format(model_name.replace("/", "-"), learning_rate, n_epoch)
    model_args.overwrite_output_dir = True
    model_args.save_best_model = True
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 100

    model = ClassificationModel(model_type, model_name, num_labels=2, args=model_args)

    cardiffnlp_datasets = pd.read_pickle(compare_datasets)
    cardiffnlp_datasets = cardiffnlp_datasets.rename(columns={"label": "labels"})
    cardiffnlp_datasets = cardiffnlp_datasets[cardiffnlp_datasets['split'] != 'test']

    columns = ["text", "labels"]

    train_df = pd.read_csv(train_file, sep="\t")
    train_df = train_df.rename(columns={"sentence": "text"}).sample(frac=1)
    if include == True:
        train_df = pd.concat([train_df[columns], cardiffnlp_datasets[columns]])
    else:
        train_df = pd.concat([cardiffnlp_datasets[columns]])


    eval_df = pd.read_csv(test_file, sep="\t")
    eval_df = eval_df.rename(columns={"sentence": "text"}).sample(frac=1)

    model.train_model(train_df=train_df, eval_df=eval_df)



model_name_to_use = sys.argv[1]
learning_rate_to_use = float(sys.argv[2])
n_epoch_to_use = int(sys.argv[3])
model_type_to_use = sys.argv[4]
include_hatemoderate = sys.argv[5]
train_hate_model(model_name_to_use, learning_rate=learning_rate_to_use, n_epoch=n_epoch_to_use, model_type= model_type_to_use, include = include_hatemoderate)
# Example: python fine_tune.py "roberta-base" 5e-6 3 "roberta" False