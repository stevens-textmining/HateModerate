# Conda env: hatemoderate
import openai
import os


openai.api_key = os.getenv('OPENAI_API_KEY')
if openai.api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")


def process_file(input_path, hate_output_path, nonhate_output_path):
    fin = open(input_path, "r")
    fin.readline()

    hate_fout = open(hate_output_path, "w")
    nonhate_fout = open(nonhate_output_path, "w")
    line_count = 0

    for line in fin:
        if line_count % 100 == 0:
            print(line_count)
        line_count += 1
        tokens = line.strip("\n").split("\t")
        example_id = tokens[0]
        sent = tokens[1]
        label = tokens[-1]
        response = openai.Moderation.create(input = sent, model="text-moderation-latest")
        toxic_score = response["results"][0]["category_scores"]["hate"]
        response = response["results"][0]["categories"]
        hate = response["hate"] or response["hate/threatening"]
        violence = response["violence"] or response["violence/graphic"]
        output_line = str(example_id) + "\t" + str(hate) + "\t" + str(violence) + "\t" + str(toxic_score) + "\n"

        if label == '1':
            hate_fout.write(output_line)
        elif label == '0':
            nonhate_fout.write(output_line)
        else:
            print(f"Invalid label: {label} on line {line_count}")

    fin.close()
    hate_fout.close()
    nonhate_fout.close()

input_path = "fine_tune/datasets/testing/hatemoderate_test.csv"
hate_output_path = "test_sota/openai/openai_score_hate.csv"
nonhate_output_path = "test_sota/openai/openai_score_nonhate.csv"

process_file(input_path, hate_output_path, nonhate_output_path)