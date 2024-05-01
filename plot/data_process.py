import pandas
import statistics
import math
import numpy as np

def mean_toxity_score(dataset_path, dataset_type="hate"):
    data = pandas.read_csv("../original/label2short.csv", delimiter="\t")
    guideline2short = {}

    for x in range(len(data)):
        guideline2short[data["guideline"][x]] = data["short"][x]

    cate2tier = {}
    tier2cate = {}

    perspective = "perspective"
    openai = "openai_score"
    fb = "fb_score"
    cardiffnlp = 'cardiffnlp_score'

    fin = open("../original/cate2guidelines.csv", "r")
    fin.readline()
    for line in fin:
        tokens = line.split("\t")
        tier = tokens[0]
        cate = tokens[1]
        cate2tier[cate] = tier
        tier2cate.setdefault(tier, [])
        tier2cate[tier].append(cate)

    fin = open(dataset_path, "r")

    fin.readline()
    cate2count = {}

    example2cate = {}
    for line in fin:
        tokens = line.strip("\n").split("\t")
        example_id = tokens[0]
        # source = tokens[2]
        cate = tokens[2]
        if cate == 'attacking intellectual capabili':
            cate = 'attacking intellectual capability'
        if cate == 'attack concept associated prote':
            cate = 'attack concept associated protected characteristics'
        if cate == 'contempt self admission intoler':
            cate = 'contempt self admission intolerance'
        tier = cate2tier[cate]
        example2cate[example_id] = cate
        cate2count.setdefault(cate, 0)
        cate2count[cate] += 1

    tier2cate2scores = {}
    suffix = "_hate.csv" if dataset_type == "hate" else "_nonhate.csv"

    for method in [perspective, openai, fb, cardiffnlp]:
        with open(method + suffix, "r") as fin:

            for line in fin:
                tokens = line.strip("\n").split("\t")
                example_id = tokens[0]
                cate = example2cate[example_id]
                tier = cate2tier[cate]
                tier2cate2scores.setdefault(tier, {})
                tier2cate2scores[tier].setdefault(cate, {})
                if method == perspective:
                    score = float(tokens[1])

                    tier2cate2scores[tier][cate].setdefault(method, {})
                    tier2cate2scores[tier][cate][method][example_id] = float(score)
                elif method == openai:
                    score1 = float(tokens[3])

                    tier2cate2scores[tier][cate].setdefault(method, {})
                    tier2cate2scores[tier][cate][method][example_id] = float(score1)

                elif method == fb:
                    score1 = float(tokens[2])

                    tier2cate2scores[tier][cate].setdefault(method, {})
                    tier2cate2scores[tier][cate][method][example_id] = float(score1)
                elif method == cardiffnlp:
                    score1 = float(tokens[2])

                    tier2cate2scores[tier][cate].setdefault(method, {})
                    tier2cate2scores[tier][cate][method][example_id] = float(score1)


    mean_scores_by_tier = {}

    for tier in ['1', '2', '3', '4']:
        cates = tier2cate[tier]
        sum_perspective = 0
        sum_openai = 0
        sum_fb = 0
        sum_cardiffnlp = 0
        total_examples = 0

        for cate in cates:
            num_examples = cate2count[cate]
            sum_perspective += sum(tier2cate2scores[tier][cate][perspective].values())
            sum_openai += sum(tier2cate2scores[tier][cate][openai].values())
            sum_fb += sum(tier2cate2scores[tier][cate][fb].values())
            sum_cardiffnlp += sum(tier2cate2scores[tier][cate][cardiffnlp].values())

            total_examples += num_examples

        mean_perspective = sum_perspective / total_examples
        mean_openai = sum_openai / total_examples
        mean_fb = sum_fb / total_examples
        mean_cardiffnlp = sum_cardiffnlp / total_examples
        mean = (sum_perspective + sum_openai + sum_fb + sum_cardiffnlp) / (4 * total_examples)
        print(f"{tier}:{mean}")

        mean_scores_by_tier[tier] = {
            perspective: mean_perspective,
            openai: mean_openai,
            fb: mean_fb,
            cardiffnlp: mean_cardiffnlp
        }

    return mean_scores_by_tier




def get_all_failrates(dataset_path, data_labels = "../original/label2short.csv", dataset_type="hate", data_guideline = "../original/cate2guidelines.csv", counter_hate = False):
    data = pandas.read_csv(data_labels, delimiter="\t")
    guideline2short = {}

    for x in range(len(data)):
        guideline2short[data["guideline"][x]] = data["short"][x]

    cate2tier = {}
    tier2cate = {}

    perspective = "perspective"
    perspective_bar = "perspective_bar"
    openai = "openai_score"
    fb = "fb_score"
    cardiffnlp = 'cardiffnlp_score'

    fin = open(data_guideline, "r")
    fin.readline()
    for line in fin:
        tokens = line.split("\t")
        tier = tokens[0]
        cate = tokens[1]
        cate2tier[cate] = tier
        tier2cate.setdefault(tier, [])
        tier2cate[tier].append(cate)

    fin = open(dataset_path, "r")
    fin.readline()
    cate2count = {}

    example2cate = {}
    for line in fin:
        tokens = line.strip("\n").split("\t")
        example_id = tokens[0]
        # source = tokens[2]
        cate = tokens[2] if counter_hate != True else tokens[7]
        if cate == 'attacking intellectual capabili':
            cate = 'attacking intellectual capability'
        if cate == 'attack concept associated prote':
            cate = 'attack concept associated protected characteristics'
        if cate == 'contempt self admission intoler':
            cate = 'contempt self admission intolerance'
        # tier = cate2tier[cate]
        example2cate[example_id] = cate
        cate2count.setdefault(cate, 0)
        cate2count[cate] += 1

    tier2cate2scores = {}
    suffix = "_hate.csv" if dataset_type == "hate" else "_nonhate.csv"

    count_error = 0
    for method in [perspective, openai, fb, cardiffnlp]:
        with open(method + suffix, "r") as fin:
            for line in fin:
                try:
                    tokens = line.strip("\n").split("\t")
                    example_id = tokens[0]
                    cate = example2cate[example_id]
                    tier = cate2tier[cate]
                    tier2cate2scores.setdefault(tier, {})
                    tier2cate2scores[tier].setdefault(cate, {})
                    if dataset_type != "hate":
                        if method == perspective:
                            score1 = float(tokens[1]) < 0.5  # non hate
                            tier2cate2scores[tier][cate].setdefault(method, {})
                            tier2cate2scores[tier][cate][method][example_id] = float(score1)
                            score2 = float(tokens[1]) < 0.7  # non hate
                            tier2cate2scores[tier][cate].setdefault(perspective_bar, {})
                            tier2cate2scores[tier][cate][perspective_bar][example_id] = float(score2)
                        elif method == openai:
                            score1 = tokens[1] == "False" or tokens[2] == "False"
                            score2 = tokens[1] == "False"
                            tier2cate2scores[tier][cate].setdefault(openai + "1", {})
                            tier2cate2scores[tier][cate][openai + "1"][example_id] = score1
                            tier2cate2scores[tier][cate].setdefault(openai + "2", {})
                            tier2cate2scores[tier][cate][openai + "2"][example_id] = score2
                        elif method == fb:
                            source = tokens[2]
                            score1 = tokens[1] == "nothate"
                            tier2cate2scores[tier][cate].setdefault(method, {})
                            tier2cate2scores[tier][cate][method][example_id] = score1
                        elif method == cardiffnlp:
                            source = tokens[2]
                            score1 = tokens[1] == "NOT-HATE"
                            tier2cate2scores[tier][cate].setdefault(method, {})
                            tier2cate2scores[tier][cate][method][example_id] = score1
                    else:
                        if method == perspective:
                            score1 = float(tokens[1]) >= 0.5  # hate
                            tier2cate2scores[tier][cate].setdefault(method, {})
                            tier2cate2scores[tier][cate][method][example_id] = float(score1)
                            score2 = float(tokens[1]) >= 0.7  # hate
                            tier2cate2scores[tier][cate].setdefault(perspective_bar, {})
                            tier2cate2scores[tier][cate][perspective_bar][example_id] = float(score2)
                        elif method == openai:
                            score1 = tokens[1] == "True" or tokens[2] == "True"
                            score2 = tokens[1] == "True"
                            tier2cate2scores[tier][cate].setdefault(openai + "1", {})
                            tier2cate2scores[tier][cate][openai + "1"][example_id] = score1
                            tier2cate2scores[tier][cate].setdefault(openai + "2", {})
                            tier2cate2scores[tier][cate][openai + "2"][example_id] = score2
                        elif method == fb:
                            source = tokens[2]
                            score1 = tokens[1] == "hate"
                            tier2cate2scores[tier][cate].setdefault(method, {})
                            tier2cate2scores[tier][cate][method][example_id] = score1
                        elif method == cardiffnlp:
                            source = tokens[2]
                            score1 = tokens[1] == "HATE"
                            tier2cate2scores[tier][cate].setdefault(method, {})
                            tier2cate2scores[tier][cate][method][example_id] = score1
                except:
                    print(tokens)
                    count_error +=1
                    print(count_error)

    all_fb = []
    all_cardiffnlp = []
    all_openai = []
    all_google = []
    all_google_bar = []
    all_labels = []

    tiers = ['1', '2', '3', '4'] if counter_hate != True else ['1']

    for tier in tiers:
        # print(tier)
        cates = tier2cate[tier]
        # print(cates)
        for cate in cates:
            short_name = guideline2short[cate]
            perspective_val = 1.0 - sum(tier2cate2scores[tier][cate][perspective].values()) / cate2count[cate]
            # print(f"total {cate} {cate2count[cate]}")
            # print(sum(tier2cate2scores[tier][cate][perspective].values()))
            if perspective_val < 0:
                raise Exception(tier2cate2scores[tier][cate][perspective], cate2count[cate])

            perspective_bar_val = 1.0 - sum(tier2cate2scores[tier][cate][perspective_bar].values()) / cate2count[cate]
            if perspective_bar_val < 0:
                raise Exception(tier2cate2scores[tier][cate][perspective_bar], cate2count[cate])

            openai2_val = 1.0 - sum(tier2cate2scores[tier][cate][openai + "2"].values()) / cate2count[cate]

            fb_val = 1.0 - sum(tier2cate2scores[tier][cate][fb].values()) / cate2count[cate]

            cardiffnlp_val = 1.0 - sum(tier2cate2scores[tier][cate][cardiffnlp].values()) / cate2count[cate]

            all_labels.append(short_name)
            all_fb.append(fb_val)
            all_cardiffnlp.append(cardiffnlp_val)
            all_openai.append(openai2_val)
            all_google.append(perspective_val)
            all_google_bar.append(perspective_bar_val)
    return all_labels, [all_fb, all_cardiffnlp, all_google, all_google_bar, all_openai]


def mean_std(list):
    mean = sum(list) / len(list)
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in list) / len(list))
    return mean, std_dev

def calculate_single_mean(data):
    flattened_data = [x for sublist in data for x in sublist if x != 0]  # Flatten the list and remove zeros
    if flattened_data:  # Check if list is not empty
        mean_value = np.mean(flattened_data)
        return mean_value

def compute_mean_failrates_per_tier(all_data):
    ed1 = 14
    ed2 = 28

    # Split the data by tiers
    t1_data = [all_data[x][:ed1] for x in range(5)]
    t2_data = [all_data[x][ed1:ed2] for x in range(5)]
    t3_data = [all_data[x][ed2:] for x in range(5)]

    # Further split tier 3 data and merge with tier 2 and 4
    t4_data = [t3_data[i][10:13] for i in range(5)]
    t2_data = [t2_data[i] + t3_data[i][:5] for i in range(5)]
    t3_data = [t3_data[i][5:10] for i in range(5)]


    means_t1 = [statistics.mean(data) for data in t1_data]
    means_t2 = [statistics.mean(data) for data in t2_data]
    means_t3 = [statistics.mean(data) for data in t3_data]
    means_t4 = [statistics.mean(data) for data in t4_data]


    models = ['FB', 'CardiffNLP', 'Google Perspective', 'Google Perspective*', 'OpenAI']
    tiers = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']

    for model, mean_t1, mean_t2, mean_t3, mean_t4 in zip(models, means_t1, means_t2, means_t3, means_t4):
        print(f"Mean failure rate for {model}:")
        for tier, mean in zip(tiers, [mean_t1, mean_t2, mean_t3, mean_t4]):
            print(f"{tier}: {mean:.2f}")
        print()

    print("Average failure rates across models:")
    selected_models = ['FB', 'CardiffNLP', 'Google Perspective', 'OpenAI']
    indices = [models.index(model) for model in selected_models]

    for tier, all_means in zip(tiers, [means_t1, means_t2, means_t3, means_t4]):
        avg_failrate = np.mean([all_means[i] for i in indices])
        print(f"{tier}: {avg_failrate:.2f}")
    print()

