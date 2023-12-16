from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import re
import pandas as pd
from scipy import stats
import torch


def predict_hate_label(model_path, file_name, dataset, device, BATCH_SIZE=32):

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    fout = open('fine_tune/testing/ttest_tmp/'+ file_name, "w")
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)
    BATCH_SIZE = BATCH_SIZE
    num_batches = len(dataset) // BATCH_SIZE + (1 if len(dataset) % BATCH_SIZE != 0 else 0)

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        batched_texts = dataset['text'][start_idx:end_idx].tolist()
        batched_labels = dataset['label'][start_idx:end_idx].tolist()

        batched_predictions = pipe(batched_texts)

        for i, (pred, actual_label, text) in enumerate(zip(batched_predictions, batched_labels, batched_texts)):
            hate = pred["label"]
            toxic_score = pred["score"] if hate == 'HATE' else 1 - pred["score"]
            fout.write(
                str(start_idx + i) + "\t" + str(hate) + "\t" + str(actual_label) + "\t" + str(toxic_score)+ "\t" + str(text) + "\n")


def generate_predictions(dataset_name, model_path, comparison_value, device):
    if dataset_name == "fine_tune/datasets/testing/Hate_Check.csv":  # local csv
        df = pd.read_csv(dataset_name, sep=',')
        df = df.rename(columns={"test_case": "text"})
        # df = df[df['split'] == 'train']
        df.drop_duplicates(subset=['text'], inplace=True)
        positive_sample = df[df['label_gold'] == 'hateful']  # .sample(1000, random_state=42)
        negative_sample = df[df['label_gold'] == 'non-hateful']  # .sample(1000, random_state=42)
        positive_sample['label'] = 'HATE'
        negative_sample['label'] = 'NOT-HATE'
        combined_sample = pd.concat([positive_sample, negative_sample]).reset_index(drop=True)
    else:  # local csv
        df = pd.read_csv(dataset_name, sep='\t')
        positive_sample = df[df['labels'] == 1]
        negative_sample = df[df['labels'] == 0]
        positive_sample['label'] = 'HATE'
        negative_sample['label'] = 'NOT-HATE'
        combined_sample = pd.concat([positive_sample, negative_sample]).reset_index(drop=True)

    match = re.search(r'/([^/]+?)(\.csv)?$', dataset_name)

    if match:
        extracted_name = match.group(1)
    else:
        print("Pattern not found in string")

    file_name = "{}_{}.csv".format(extracted_name, model_path.split('/')[-1])


    predict_hate_label(model_path, file_name, combined_sample, device)



def ttest_all(prediction_A, prediction_B):
    predictions_model_A = pd.read_csv('fine_tune/testing/ttest_tmp/' + prediction_A, sep='\t', usecols=[0, 1, 2], header=None,
                                      names=['id', 'prediction', 'actual'])
    predictions_model_B = pd.read_csv('fine_tune/testing/ttest_tmp/' + prediction_B, sep='\t', usecols=[0, 1, 2], header=None,
                                      names=['id', 'prediction', 'actual'])

    predictions_model_A['prediction'] = predictions_model_A['prediction'].map({'LABEL_1': 1, 'LABEL_0': 0})
    predictions_model_A['actual'] = predictions_model_A['actual'].map({'HATE': 1, 'NOT-HATE': 0})

    predictions_model_B['prediction'] = predictions_model_B['prediction'].map({'LABEL_1': 1, 'LABEL_0': 0})
    predictions_model_B['actual'] = predictions_model_B['actual'].map({'HATE': 1, 'NOT-HATE': 0})

    accuracy_A = (predictions_model_A['prediction'] == predictions_model_A['actual']).mean()
    accuracy_B = (predictions_model_B['prediction'] == predictions_model_B['actual']).mean()

    t_statistic, p_value = stats.ttest_rel(predictions_model_A['prediction'], predictions_model_B['prediction'])

    print("Model A Accuracy:", accuracy_A)
    print("Model B Accuracy:", accuracy_B)
    print("t-statistic:", t_statistic)
    print("p-value:", p_value)


def ttest(prediction_A, prediction_B):
    predictions_model_A = pd.read_csv('fine_tune/testing/ttest_tmp/' + prediction_A, sep='\t', usecols=[0, 1, 2], header=None,
                                      names=['id', 'prediction', 'actual'])
    predictions_model_B = pd.read_csv('fine_tune/testing/ttest_tmp/' + prediction_B, sep='\t', usecols=[0, 1, 2], header=None,
                                      names=['id', 'prediction', 'actual'])

    predictions_model_A['prediction'] = predictions_model_A['prediction'].map({'LABEL_1': 1, 'LABEL_0': 0})
    predictions_model_A['actual'] = predictions_model_A['actual'].map({'HATE': 1, 'NOT-HATE': 0})
    predictions_model_B['prediction'] = predictions_model_B['prediction'].map({'LABEL_1': 1, 'LABEL_0': 0})
    predictions_model_B['actual'] = predictions_model_B['actual'].map({'HATE': 1, 'NOT-HATE': 0})

    predictions_A_hate = ((predictions_model_A['prediction'] == 1) & (predictions_model_A['actual'] == 1)).astype(int)
    predictions_B_hate = ((predictions_model_B['prediction'] == 1) & (predictions_model_B['actual'] == 1)).astype(int)

    predictions_A_nonhate = ((predictions_model_A['prediction'] == 0) & (predictions_model_A['actual'] == 0)).astype(
        int)
    predictions_B_nonhate = ((predictions_model_B['prediction'] == 0) & (predictions_model_B['actual'] == 0)).astype(
        int)

    accuracy_A_hate = (predictions_model_A[predictions_model_A['actual'] == 1]['prediction'] ==
                       predictions_model_A[predictions_model_A['actual'] == 1]['actual']).mean()
    accuracy_B_hate = (predictions_model_B[predictions_model_B['actual'] == 1]['prediction'] ==
                       predictions_model_B[predictions_model_B['actual'] == 1]['actual']).mean()

    accuracy_A_nonhate = (predictions_model_A[predictions_model_A['actual'] == 0]['prediction'] ==
                          predictions_model_A[predictions_model_A['actual'] == 0]['actual']).mean()
    accuracy_B_nonhate = (predictions_model_B[predictions_model_B['actual'] == 0]['prediction'] ==
                          predictions_model_B[predictions_model_B['actual'] == 0]['actual']).mean()

    #     t_statistic_hate, p_value_hate = stats.ttest_rel(predictions_A_hate, predictions_B_hate)
    #     t_statistic_nonhate, p_value_nonhate = stats.ttest_rel(predictions_A_nonhate, predictions_B_nonhate)
    t_statistic_hate, p_value_hate = stats.ttest_rel(predictions_A_hate, predictions_B_hate)
    t_statistic_nonhate, p_value_nonhate = stats.ttest_rel(predictions_A_nonhate, predictions_B_nonhate)

    print("Model A Accuracy (Hate):", accuracy_A_hate)
    print("Model B Accuracy (Hate):", accuracy_B_hate)
    print("t-statistic (Hate):", t_statistic_hate)
    print("p-value (Hate):", p_value_hate)
    #     print("\n")

    print("Model A Accuracy (Non-Hate):", accuracy_A_nonhate)
    print("Model B Accuracy (Non-Hate):", accuracy_B_nonhate)
    print("t-statistic (Non-Hate):", t_statistic_nonhate)
    print("p-value (Non-Hate):", p_value_nonhate)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets=(
          "fine_tune/datasets/testing/Hate_Check.csv",
          "fine_tune/datasets/testing/hatemoderate_test.csv",
          "fine_tune/datasets/testing/test_hatE.csv",
          "fine_tune/datasets/testing/test_hatex.csv",
          "fine_tune/datasets/testing/test_htpo.csv"
          )

models=("roberta-base_lr=2e-05_epoch=4_batch_size=32_include_hatemoderate",
        "roberta-base_lr=2e-05_epoch=4_batch_size=32_no-include_hatemoderate",
       "roberta-base_lr=2e-05_epoch=4_batch_size=32_rebalance")

labels = "LABEL_1"

for dataset in datasets:
    for model in models:
        print(f"Model: {model}\n")
        generate_predictions(dataset, model, labels, device)


Rebalance = "roberta-base_lr=2e-05_epoch=4_batch_size=32_rebalance.csv"
Card = "roberta-base_lr=2e-05_epoch=4_batch_size=32_no-include_hatemoderate.csv"
HM = "roberta-base_lr=2e-05_epoch=4_batch_size=32_include_hatemoderate.csv"

models_name = ['Rebalance', 'Card', 'HM']
models = [Rebalance, Card, HM]
test_cases = ['Hate_Check', 'hatemoderate_test', 'test_hatE', 'test_hatex', 'test_htpo']

for test in test_cases:
    print(f"Test Case: {test}")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_A_file = f"{test}_{models[i]}"
            model_B_file = f"{test}_{models[j]}"

            print(f"Comparing {models_name[i]} vs {models_name[j]}")

            ttest_all(model_A_file, model_B_file)
            ttest(model_A_file, model_B_file)
            print("\n")