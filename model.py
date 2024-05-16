import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Load pre-trained BERT model and tokenizer
bertBase = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bertBase)
model = BertForSequenceClassification.from_pretrained(bertBase)

# Function to score arguments
def score(text):
    encoded_inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**encoded_inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1).detach().numpy()
    
    return probabilities[:, 1].item()

dataset_path = './data/arg_quality_rank_30k.csv'
df = pd.read_csv(dataset_path)

arguments = df['argument'].tolist()
actual_scores = df['WA'].tolist()

arguments_train, arguments_test, scores_train, scores_test = train_test_split(
    arguments, actual_scores, test_size=0.2, random_state=42
)

# print("Prediction occurring")
# predicted_scores_test = [score(arg) for arg in arguments_test]
# print("Prediction complete")

# plt.scatter(scores_test, predicted_scores_test)
# plt.xlabel('Actual QS')
# plt.ylabel('Predicted QS')
# plt.title('Predicted vs Actual QS')
# plt.show()