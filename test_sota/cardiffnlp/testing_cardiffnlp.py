#conda env: hatemoderate
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

def process_file(input_path, output_path):
    with open(input_path, "r") as fin:
        fin.readline()
        with open(output_path, "w") as fout:
            line_count = 0

            for line in fin:
                if line_count % 100 == 0:
                    print(line_count)
                line_count += 1
                tokens = line.strip("\n").split("\t")
                if len(tokens) < 5:
                    print(f"Skipping line {tokens[0]}: {line}")
                    continue

                example_id = tokens[0]
                sent = tokens[1]
                source = tokens[2]
                cate = tokens[4]

                hate = pipe(sent)[0]["label"]
                toxic_score = pipe(sent)[0]["score"] if hate == 'HATE' else 1 - pipe(sent)[0]["score"]
                fout.write(str(example_id) + "\t" + str(hate) + "\t" + str(toxic_score) + "\t" + source + "\n")

    fout.close()


hate_input_path = "../postprocess/hate_test.csv"
hate_output_path = "cardiffnlp_score.csv"

nonhate_input_path = "../postprocess/nonhate_test.csv"
nonhate_output_path = "cardiffnlp_score_nonhate.csv"

process_file(hate_input_path, hate_output_path)
process_file(nonhate_input_path, nonhate_output_path)