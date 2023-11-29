import torch
from utils import predict_hate_label, calculate_matching
from datasets import load_dataset
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_hatecheck_dataset(model_path, comparison_value, device, target_ident_column):
    dataset = load_dataset("Paul/hatecheck")
    df = dataset['test'].to_pandas()
    df = df.rename(columns={"test_case": "text"})
    df.drop_duplicates(subset=['text'], inplace=True)
    df['label'] = df['label_gold'].apply(lambda x: 'HATE' if x == 'hateful' else 'NOT-HATE')

    # Predict hate labels
    file_name = "hatecheck_" + model_path.split('/')[-1] + ".csv"
    predict_hate_label(model_path, file_name, df, device)

    # Load predictions
    df_predictions = pd.read_csv('fine_tune/tmp/' + file_name, sep='\t', header=None)

    # Filter for NOT-HATE cases and group by target_ident
    df_not_hate = df_predictions[df_predictions[2] == 'NOT-HATE']
    grouped = df_not_hate.groupby(df[target_ident_column])  # Use the actual column name for target_ident

    # Calculate accuracy for each group
    accuracy_per_group = {}
    for group_name, group_df in grouped:
        matching, total = calculate_matching(group_df, 'NOT-HATE', comparison_value)
        accuracy = (matching / total) * 100
        accuracy_per_group[group_name] = accuracy

    return accuracy_per_group


models = (
"roberta-base_lr=2e-05_epoch=4_batch_size=32_include_hatemoderate/checkpoint-15304-epoch-2",
"roberta-base_lr=2e-05_epoch=4_batch_size=32_no-include_hatemoderate/checkpoint-15162-epoch-2",
)
comparison_value = "LABEL_1"
match_column = "target_ident"

for model in models:
    print(f"Model: {model}\n")
    accuracies = process_hatecheck_dataset(model, comparison_value, device, match_column)
    print(accuracies)