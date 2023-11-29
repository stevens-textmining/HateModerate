#conda env: hatemoderate

import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r1-target")
tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r1-target")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)


def process_file(input_path, hate_output_path, nonhate_output_path):
    with open(input_path, "r") as fin:
        fin.readline()
        hate_fout = open(hate_output_path, "w")
        nonhate_fout = open(nonhate_output_path, "w")
        line_count = 0

        for line in fin:
            if line_count % 100 == 0:
                print(line_count)
            line_count += 1
            tokens = line.strip("\n").split("\t")
            if len(tokens) < 4:
                print(f"Skipping line {tokens[0]}: {line}")
                continue

            example_id = tokens[0]
            sent = tokens[1]
            source = tokens[2]
            label = tokens[-1]

            hate = pipe(sent)[0]["label"]
            toxic_score = pipe(sent)[0]["score"] if hate == 'hate' else 1 - pipe(sent)[0]["score"]
            output_line = str(example_id) + "\t" + str(hate) + "\t" + str(toxic_score) + "\t" + source + "\n"

            if label == '1':
                hate_fout.write(output_line)
            elif label == '0':
                nonhate_fout.write(output_line)
            else:
                print(f"Invalid label: {label} on line {line_count}")

    hate_fout.close()
    nonhate_fout.close()

input_path = "fine_tune/datasets/testing/hatemoderate_test.csv"
hate_output_path = "test_sota/facebook/fb_score_hate.csv"
nonhate_output_path = "test_sota/facebook/fb_score_nonhate.csv"

process_file(input_path, hate_output_path, nonhate_output_path)