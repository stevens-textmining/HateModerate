# Examining AI System Behaviors against Rule-based Requirements: A Case Study on AI-based Content Moderation

This is the anonymous repository for our paper "Examining AI System Behaviors against Rule-based Requirements: A Case Study on AI-based Content Moderation"

## Install
```
pip install torch pandas transformers datasets simpletransformers tqdm googleapiclient openai
```

Dependencies:

- [pytorch](https://pytorch.org)
-  `pandas` for loading and processing csv files
-  `transformers` for huggingface transformers
-  `datasets` for huggingface datasets
-  `tqdm` for progress bars
-  `simpletransformers` for easily fine-tuning training
-  `googleapiclient` for accessing Google's Perspective API
-  `openai` for accessing OpenAI's API

## Aggregating the data annotated by annotators:

0. Aggregating the annotators' labels: t1.xlsx...t4.xlsx -> Aggregation.py -> all_data.csv
1. Get the hatemoderate dataset: all_data.csv, all_examples.csv (from the last submission) -> remove_majority_vote.py -> all_examples_new.csv 

## Step 2 (Section IV) Examining AI-based Content Moderation Software against Policies

### Instructions for reproducing (requires API keys for OpenAI and Google Perspective) 

0. Testing facebook's API using hatemoderate: all_examples_new_hate.csv -> testing_fb.py -> fb.csv
1. Testing OpenAI's API using hate moderate: all_examples_new_hate.csv -> testing_openai.py -> openai.csv
2. Testing Google's Perspective API using hatemoderate: all_examples_new_hate.csv -> testing_perspective.py -> perspective.csv
3. Plotting the testing results as Figure 6: fb.csv openai.csv perspective.csv -> get_testing_result.py -> app1.png (Figure 6)

### Main results

The main results of Step 2 can be found in the image below. It shows the failure rates detected for each of the 41 policies in Facebook's community standards. From left to right: Facebook's RoBERTa model finetuned on the DynaHate dataset tested with the entire HateModerate dataset, the same RoBERTa model tested with non-DynaHate cases in HateModerate, Google's Perspective API, and OpenAI's Moderation API (moderation-latest).


<p align="center">
    <img src="https://anonymous.4open.science/r/hatemoderate-835F/hatemoderate/step2/app1.png"  width=1000>
    <br>
</p>


## Step 3 (Section V) Automatically Matching Examples to Policies 

### Instructions for reproducing (requires API keys for OpenAI)

0. Split hatemoderate into train and test: all_examples_new_hate.csv -> split_train_test.py train.csv test.csv
1. Convert train into OpenAI compatible format: train.csv -> get_train.py -> train_pr0.jsonl (convert train to OpenAI compatible format)
2. Prepare OpenAI training data: train_pr0.jsonl -> prepare_openai_compatible.sh -> train_pr0_prepared.jsonl
3. Finetune using OpenAI API train_pr0_prepared.jsonl -> finetune.sh -> wait until finish to get the model name (Finetuned OpenAI model (models with stevens...))
4. Use OpenAI API to predict test and compare with the ground truth to get the prediction accuracy: test.csv, model name -> prediction.py -> classification accuracy pasted to "experiment" spreadsheet (create two new tabs)
5. Update table I
6. Update Figure 7 using plot_step3.py

### Main Results

In Step 3, we match the examples to one of the policies. The overall accuracies can be found in the table below:

<p align="center">
    <img src="https://anonymous.4open.science/r/hatemoderate-835F/hatemoderate/step3/app2_table.png"  width=500>
    <br>
</p>

The results of varying the hyperparameter in Step 3 can be found below:


<p align="center">
    <img src="https://anonymous.4open.science/r/hatemoderate-835F/hatemoderate/step3/app2.png"  width=1300>
    <br>
</p>



Testing SOTA softwares
python test_sota/cardiffnlp/testing_cardiffnlp.py
python test_sota/facebook/testing_fb.py
python test_sota/google/testing_perspective.py
python test_sota/google/testing_openai.py

Fine-tune LLMs
python fine_tune/training/fine_tune.py "roberta-base" 5e-6 3 "roberta" False
./fine_tune/training/run_fine_tune.sh

Test finetuned model
python fine_tune/testing/test.py
