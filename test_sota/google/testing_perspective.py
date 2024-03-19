# conda env: hatemoderate

from googleapiclient import discovery
from googleapiclient.errors import HttpError
import time
import os


API_KEY = os.environ.get('GOOGLE_API_KEY')
if API_KEY is None:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)


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

        analyze_request = {
            'comment': {'text': sent},
            'requestedAttributes': {'TOXICITY': {}}
        }

        try:
            response = client.comments().analyze(body=analyze_request).execute()
            toxic_score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            output_line = str(example_id) + "\t" + str(toxic_score) + "\t" + str(label) + "\n"

            if label == '1':
                hate_fout.write(output_line)
            elif label == '0':
                nonhate_fout.write(output_line)
            else:
                print(f"Invalid label: {label} on line {line_count}")

            hate_fout.flush()
            nonhate_fout.flush()
            time.sleep(1)
        except HttpError as er:
            print("Error encountered. Sleeping for 5 seconds before retrying.")
            time.sleep(5)
            pass

    fin.close()
    hate_fout.close()
    nonhate_fout.close()


input_path = "fine_tune/datasets/testing/hatemoderate_test.csv"
hate_output_path = "test_sota/google/perspective_hate.csv"
nonhate_output_path = "test_sota/google/perspective_nonhate.csv"

process_file(input_path, hate_output_path, nonhate_output_path)