# conda env: hatemoderate

from googleapiclient import discovery
from googleapiclient.errors import HttpError
import time

API_KEY = 'YOUR_API_KEY'

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)


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

        analyze_request = {
            'comment': {'text': sent},
            'requestedAttributes': {'TOXICITY': {}}
        }

        try:
            response = client.comments().analyze(body=analyze_request).execute()
            toxic_score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            fout.write(str(example_id) + "\t" + str(toxic_score) + "\n")
            fout.flush()
            time.sleep(1)
        except HttpError as er:
            print("Error encountered. Sleeping for 5 seconds before retrying.")
            time.sleep(5)
            pass

    fout.close()



hate_input_path = "../postprocess/hate_test.csv"
hate_output_path = "perspective.csv"

nonhate_input_path = "../postprocess/nonhate_test.csv"
nonhate_output_path = "perspective_nonhate.csv"

process_file(hate_input_path, hate_output_path)
process_file(nonhate_input_path, nonhate_output_path)