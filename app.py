from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

bertBase = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bertBase)
model = AutoModelForSequenceClassification.from_pretrained(bertBase)

def score_arg(text):

    text = ' '.join(text.split())

    if len(text) < 10:
        return 0.0

    encoded_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding='max_length', add_special_tokens=True)
    outputs = model(**encoded_inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1).detach().numpy()

    return probabilities[:, 1].item()

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def score():
    data = request.form['text']
    score = score_arg(data)
    
    score = round(100 * score, 2)

    return jsonify({'Argument': data, 'Quality Score': score})

if __name__ == "__main__":
    app.run(debug=True)