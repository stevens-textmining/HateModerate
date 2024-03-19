from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from transformers import pipeline
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models paths
models_paths = [
    "/data/jzheng36/HateModerate/roberta-base_lr=2e-05_epoch=4_batch_size=32_include_hatemoderate/checkpoint-15304-epoch-2",
    "/data/jzheng36/HateModerate/roberta-base_lr=2e-05_epoch=4_batch_size=32_no-include_hatemoderate/checkpoint-15162-epoch-2",
    "/data/jzheng36/HateModerate/best_hp_rebalance"
]

# Initialize models and pipelines
pipelines = []
for model_path in models_paths:
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)
    cardiffnlp = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-hate-latest")
    pipelines.append(pipe)

cardiffnlp = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-hate-latest")
pipelines.append(cardiffnlp)

test_sentences = (
"Being middle eastern is cool.", "Being chinese is cool.", "Being muslim is cool.", "Being gay is cool.",
"Being lgbtq is cool.", "Being homosexual is cool.", "Being black is cool.", "Being black is cool.",
"Being woman is cool.", "Being latino is cool.", "Being old is cool.", "Being immigrant is cool.",
"Being trans is cool.",
"Being Jews is cool.", "Being trans is cool.", "Being lesbian is cool.", "black women are good locking.",
"I love chinese.", "Jews are smart.", "Asian girls are cute.", "I like trans.",
"women are selfish.",
"Jews are selfish.", "Blacks are foolish.", "Chinese are selfish.", "Middle Eastern person is selfish.",
"Muslim is filthy.",
"lgbtq is filthy.", "gay is filthy.", "Muslim is filthy.",
)

for test_sentence in test_sentences:
    predictions_text = f"{test_sentence};"
    for i, pipe in enumerate(pipelines):
        prediction = pipe(test_sentence)
        label = prediction[0]['label']
        score = prediction[0]['score']

        is_hate = label == "LABEL_1"
        if i == 3:
            is_hate = label == "HATE"
        hate_probability = score if is_hate else (1 - score)
        non_hate_probability = 1 - hate_probability

        predictions_text += f"Model {i + 1}:'{hate_probability:.5f};"

    print(predictions_text)