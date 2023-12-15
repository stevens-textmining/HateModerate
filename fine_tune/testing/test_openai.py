import openai
import os
from utils import calculate_matching
import pandas as pd
import re



openai.api_key = os.getenv('OPENAI_API_KEY')
if openai.api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

def calculate_matching(df, actual_label, comparison_value):
    subset_df = df[df[2] == actual_label]
    if actual_label == 'HATE':
        matching = (subset_df[1] == comparison_value).sum()
    elif actual_label == 'NOT-HATE':
        matching = (subset_df[1] != comparison_value).sum()
    return matching, len(subset_df)


def predict_openai_label(output_path, dataset):
    with open(output_path, "w") as fout:
        line_count = 0

        for index, line in dataset.iterrows():
            if line_count % 100 == 0:
                print(line_count)
            line_count += 1
            example_id = line_count
            line = line.to_dict()
            sent = line['text']
            label = line['label']
            response = openai.Moderation.create(input = sent, model="text-moderation-latest")
            toxic_score = response["results"][0]["category_scores"]["hate"]
            response = response["results"][0]["categories"]
            hate = response["hate"] or response["hate/threatening"]
            output_line = str(example_id) + "\t" + str(hate) + "\t" + str(label) + "\t" + str(toxic_score) + "\n"
            # print(output_line)

            fout.write(output_line)


def process_dataset(dataset_name, comparison_value):
    if dataset_name == "fine_tune/datasets/testing/Hate_Check.csv":  # local csv
        df = pd.read_csv(dataset_name, sep=',')
        df = df.rename(columns={"test_case": "text"})
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

    file_name = "{}_openai.csv".format(extracted_name)


    predict_openai_label(file_name, combined_sample)
    df = pd.read_csv(file_name, sep='\t', header=None)
    matching_hate, total_hate = calculate_matching(df, 'HATE', comparison_value)
    matching_nothate, total_nothate = calculate_matching(df, 'NOT-HATE', comparison_value)
    print(matching_nothate)

    results = (f"Results for {extracted_name} using OpenAi :\n"
               f"{(matching_hate / total_hate) * 100:.2f}% of accuracy of HATE cases.\n"
               f"{(matching_nothate / total_nothate) * 100:.2f}% of accuracy of NOT-HATE cases.\n"
               f"{(matching_hate + matching_nothate) / len(df) * 100:.2f}% of accuracy of all cases.\n\n")

    return results


datasets=(
    "fine_tune/datasets/testing/Hate_Check.csv",
          "fine_tune/datasets/testing/hatemoderate_test.csv",
          "fine_tune/datasets/testing/test_hatE.csv",
          "fine_tune/datasets/testing/test_hatex.csv",
          "fine_tune/datasets/testing/test_htpo.csv"
          )

# datasets=(
#     "fine_tune/datasets/testing/test.csv",
#           )

labels = True
output_file = "test_openai_results.txt"

with open(output_file, "a") as f:
    for dataset in datasets:
        f.write(f"Dataset: {dataset}\n")
        print(f"Processing Dataset: {dataset}")  # This will print to the terminal.

        results = process_dataset(dataset, labels)
        print(results)
        f.write(results)