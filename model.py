from transformers import BertTokenizer, BertForSequenceClassification
import torch

bertBase = "bert-base-uncased" # bert-large is too expensive
tokenizer = BertTokenizer.from_pretrained(bertBase)
model = BertForSequenceClassification.from_pretrained(bertBase, num_labels=1)

def score(text):

    inputs = tokenizer(text, do_lower_case=True, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Output processing
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

    # Positive class probability
    return probabilities[0][1].item()