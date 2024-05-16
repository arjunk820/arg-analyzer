from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

bertBase = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bertBase)
model = BertForSequenceClassification.from_pretrained(bertBase)

def evaluate(text):

    text = ' '.join(text.split())

    if len(text) < 10:
        return 0.0

    encoded_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

    with torch.no_grad(): 
        outputs = model(**encoded_inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    return probabilities[:, 1].item()

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def score():
    data = request.form['text']
    score = evaluate(data)
    
    score = round(100 * score, 2)

    return jsonify({'Argument': data, 'Quality Score': score})

if __name__ == "__main__":
    app.run(debug=True)