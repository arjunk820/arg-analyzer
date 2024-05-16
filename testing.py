import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


CONFIG = {
    'bertBase': "bert-base-uncased",
    'dataset_path': './data/arg_quality_rank_30k.csv',
    'max_length': 512,
    'min_length': 16,
}

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(CONFIG['bertBase'])
model = BertForSequenceClassification.from_pretrained(CONFIG['bertBase'])

def score(text):

    """
        Function to score arguments, same as app.py
    """

    text = ' '.join(text.split())

    if len(text) < CONFIG['min_length']:
        return 0.0

    encoded_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONFIG['max_length'])
    outputs = model(**encoded_inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1).detach().numpy()
    
    return probabilities[:, 1].item()

df = pd.read_csv(CONFIG['dataset_path'])

arguments = df['argument'].tolist()
actual_scores = df['WA'].tolist()

arguments_train, arguments_test, scores_train, scores_test = train_test_split(
    arguments, actual_scores, test_size=0.2, random_state=42
)

# WA allows for more ability to score quality on a continuous scale (0-1)
predicted_scores_test = [score(arg) for arg in arguments_test]

mse = mean_squared_error(scores_test, predicted_scores_test)
print(f"Mean Squared Error: {mse}")

# Graph plot of predicted vs WAs
plt.scatter(scores_test, predicted_scores_test)
plt.xlabel('Actual QS')
plt.ylabel('Predicted QS')
plt.title('Predicted vs Actual QS')
plt.show()