# Conda env: hatemoderate 

import openai
openai.api_key = 'YOUR_API_KEY'


def process_file(input_path, output_path):
    fin = open(input_path, "r")
    fin.readline()

    fout = open(output_path, "w")
    line_count = 0

    for line in fin:
        if line_count % 100 == 0:
            print(line_count)
        line_count += 1
        tokens = line.strip("\n").split("\t")
        example_id = tokens[0]
        sent = tokens[1]
        response = openai.Moderation.create(input = sent, model="text-moderation-latest")
        toxic_score = response["results"][0]["category_scores"]["hate"]
        response = response["results"][0]["categories"]
        hate = response["hate"] or response["hate/threatening"]
        violence = response["violence"] or response["violence/graphic"]
        fout.write(str(example_id) + "\t" + str(hate) + "\t" + str(violence) + "\t" + str(toxic_score) + "\n")

    fout.close()
hate_input_path = "../postprocess/hate_test.csv"
hate_output_path = "openai_score.csv"

nonhate_input_path = "../postprocess/nonhate_test.csv"
nonhate_output_path = "openai_score_nonhate.csv"

process_file(hate_input_path, hate_output_path)
process_file(nonhate_input_path, nonhate_output_path)