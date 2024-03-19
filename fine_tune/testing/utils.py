from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from datasets import load_dataset
import pandas as pd
import re


def predict_hate_label(model_path, file_name, dataset, device, BATCH_SIZE=32):

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    fout = open('fine_tune/tmp/'+ file_name, "w")
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

def calculate_matching(df, actual_label, comparison_value):
    subset_df = df[df[2] == actual_label]
    if actual_label == 'HATE':
        matching = (subset_df[1] == comparison_value).sum()
    elif actual_label == 'NOT-HATE':
        matching = (subset_df[1] != comparison_value).sum()
    return matching, len(subset_df)

def process_dataset(dataset_name, model_path, comparison_value, device):
    if dataset_name == "hate_speech_offensive":
        dataset = load_dataset(dataset_name)
        df = dataset["train"].to_pandas()
        df.drop_duplicates(subset=['tweet'], inplace=True)
        positive_sample = df[df['class'] == 0].sample(1000, random_state=42)
        negative_sample = df[(df['class'] == 1) | (df['class'] == 2)].sample(1000, random_state=42)
        positive_sample['label'] = 'HATE'
        negative_sample['label'] = 'NOT-HATE'
        combined_sample = pd.concat([positive_sample, negative_sample]).reset_index(drop=True)
        combined_sample = combined_sample.rename(columns={"tweet": "text"})
    elif dataset_name == "hatexplain":
        dataset = load_dataset(dataset_name)
        texts = []
        labels = []
        for entry in dataset["train"]:
            label = 'HATE' if entry['annotators']['label'].count(0) >= len(
                entry['annotators']['label']) / 2 else 'NOT-HATE'
            texts.append(' '.join(entry['post_tokens']))
            labels.append(label)
        df = pd.DataFrame({'text': texts, 'label': labels}).drop_duplicates(subset=['text'])
        positive_sample = df[df['label'] == 'HATE'].sample(1000, random_state=42)
        negative_sample = df[df['label'] != 'HATE'].sample(1000, random_state=42)
        combined_sample = pd.concat([positive_sample, negative_sample]).reset_index(drop=True)
    elif dataset_name == "ucberkeley-dlab/measuring-hate-speech":
        # df = pd.read_csv(dataset_name, sep = "\t")
        # df = df[df['split'] == 0.5]
        dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
        df = dataset['train'].to_pandas()
        df.drop_duplicates(subset=['text'], inplace=True)
        positive_sample = df[df['hate_speech_score'] > 0.5].sample(1000, random_state=42)
        negative_sample = df[(df['hate_speech_score'] < 0.5) & (df['hate_speech_score'] > -5)].sample(1000,
                                                                                                      random_state=42)
        positive_sample['label'] = 'HATE'
        negative_sample['label'] = 'NOT-HATE'
        combined_sample = pd.concat([positive_sample, negative_sample]).reset_index(drop=True)
    elif dataset_name == "fine_tune/datasets/testing/Hate_Check.csv":  # local csv
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
        df = df.rename(columns={"sentence": "text"})
        df = df.rename(columns={"label": "labels"})
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
    df = pd.read_csv('fine_tune/tmp/'+ file_name, sep='\t', header=None)
    matching_hate, total_hate = calculate_matching(df, 'HATE', comparison_value)
    matching_nothate, total_nothate = calculate_matching(df, 'NOT-HATE', comparison_value)

    print(f"Results for {extracted_name} using model {model_path}:")
    print(f"{(matching_hate / total_hate) * 100:.2f}% of accuracy of HATE cases.")
    print(f"{(matching_nothate / total_nothate) * 100:.2f}% of accuracy of NOT-HATE cases.")
    print(f"{(matching_hate + matching_nothate) / len(df) * 100:.2f}% of accuracy of all cases.")
    print("\n")

