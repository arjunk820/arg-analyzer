from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

bertBase = "bert-base-uncased" # bert-large is too expensive
tokenizer = BertTokenizer.from_pretrained(bertBase)
model = BertForSequenceClassification.from_pretrained(bertBase, num_labels=2)

def score(text):

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    return probabilities[:, 1].item()

data = pd.read_csv("./data/arg_quality_rank_30k.csv")
argument = data['argument']
quality = data['WA']

train_texts, test_texts, train_labels, test_labels = train_test_split(argument, quality, test_size=0.2, random_state=42)

for text in test_texts.sample(5):  # testing with 5 random samples
    print('Text:', text)
    print('Score:', score(text), '\n')