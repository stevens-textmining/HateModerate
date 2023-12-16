#!/usr/bin/env python
import krippendorff
import numpy as np
import pandas as pd

def nonhate():
    path = "postprocess/all_examples_nonhate_raw.csv"
    df = pd.read_csv(path, sep='\t')
    df.replace(-1, np.nan, inplace=True)

    reliability_data = [df["is_valid_human_2"].tolist(), df["is_valid_human_1"].tolist()]
    print(len(reliability_data))

    try:
        print("Krippendorff's alpha for nominal metric: ",
              krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal'))
        print("Krippendorff's alpha for interval metric: ", krippendorff.alpha(reliability_data=reliability_data))
    except ValueError as e:
        print("Error calculating Krippendorff's alpha: ", e)

def hate_agreement():
    path = "all_examples_hate_raw.csv"
    df = pd.read_csv(path, sep='\t')
    df = df[df['votes'].apply(lambda x: len(x.split(';')) > 2)]

    votes_data = [list(map(int, str(votes).split(';')[:2])) for votes in df['votes']]

    total_pairs = 0
    agree_pairs = 0

    for votes in votes_data:
        # print(votes)
        if len(votes) == 2:
            total_pairs += 1
            if votes[0] == votes[1]:
                agree_pairs += 1

    agreement_rate = agree_pairs / total_pairs if total_pairs > 0 else 0
    print("Agreement rate:", agreement_rate)

def hate():
    path = "postprocess/all_examples_hate_raw.csv"
    # path = "../hatemoderate/hatemoderate/postprocess/all_examples_0601.csv"
    df = pd.read_csv(path, sep='\t')
    # df = df[df["dataset"] != 'gpt']

    # df = df[df['votes'].apply(lambda x: len(x.split(';')) > 1)]

    # df = df.sample(frac=1).reset_index(drop=True)
    votes_data = [list(map(int, str(votes).split(';')[:])) for votes in df['votes']]

    max_length = max(len(row) for row in votes_data)
    # standardized_votes = [row + [np.nan] * (max_length - len(row)) for row in votes_data]
    standardized_votes = [row + [np.nan] * (max_length - len(row)) for row in votes_data]
    votes_array = np.array(standardized_votes).T
    print(votes_array)
    print(len(votes_array[0]))

    print("Krippendorff's alpha for nominal metric: ", krippendorff.alpha(reliability_data=votes_array, level_of_measurement='nominal'))
    print("Krippendorff's alpha for interval metric: ", krippendorff.alpha(reliability_data=votes_array))



if __name__ == '__main__':
    # nonhate()
    hate_agreement()
    hate()