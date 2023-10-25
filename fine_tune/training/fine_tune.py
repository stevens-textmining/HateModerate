from compare_accuracies_models_datasets import utils
import sys

model_name_to_use = sys.argv[1]
learning_rate_to_use = float(sys.argv[2])
n_epoch_to_use = int(sys.argv[3])
model_type_to_use = sys.argv[4]
include_hatemoderate = sys.argv[5]
utils.train_hate_model(model_name_to_use, learning_rate=learning_rate_to_use, n_epoch=n_epoch_to_use, model_type= model_type_to_use, include = include_hatemoderate)
# python fine_tune.py "cardiffnlp/twitter-roberta-base-hate-latest" 5e-6 3 "roberta" False