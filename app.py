from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

bertBase = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bertBase)
model = BertForSequenceClassification.from_pretrained(bertBase)

# evaluate()
# Inputs: string
# Returns: float
# Expects: a string with text within 16-512 characters
# Notes: The function uses the tokenizer and model instantiated above
# Purpose: The function processes the text and gives a quality score

def evaluate(text):

    text = ' '.join(text.split())

    if len(text) < 16:
        return 0.0

    encoded_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

    with torch.no_grad(): 
        outputs = model(**encoded_inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    return probabilities[:, 1].item()


# home()
# Inputs: N/A
# Returns: render_template(index.html)
# Expects: index.html in proj directory
# Notes: N/A
# Purpose: This function presents index.html on running the application

@app.route("/")
def home():
    return render_template('index.html')

# score()
# Inputs: N/A
# Returns: data (text) and evaluate(data)
# Expects: input in the form in the app
# Notes: N/A
# Purpose: This function calls evaluate() and returns the input text and score in json format

@app.route('/evaluate', methods=['POST'])
def score():
    data = request.form['text']
    score = evaluate(data)
    
    # Outputting the result in % form for readability
    score = round(100 * score, 2)

    return jsonify({'Argument': data, 'Quality Score': score})

if __name__ == "__main__":
    app.run(debug=True)