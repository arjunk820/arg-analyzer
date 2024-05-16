from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

CONFIG = {
    'bertBase': "bert-base-uncased",
    'min_length': 16,
    'max_length': 512
}

tokenizer = BertTokenizer.from_pretrained(CONFIG['bertBase'])
model = BertForSequenceClassification.from_pretrained(CONFIG['bertBase'])

def evaluate(text):

    """
    
    evaluate() function
    Inputs: string
    Returns: float
    Expects: a string with text within 16-512 characters
    Notes: The function uses the tokenizer and model instantiated above
    Purpose: The function processes the text and gives a quality score
    
    """

    text = ' '.join(text.split())

    if len(text) < CONFIG['min_length']:
        return 0.0

    encoded_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONFIG['max_length'], padding=True)

    with torch.no_grad(): 
        outputs = model(**encoded_inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    return probabilities[:, 1].item()


@app.route("/")
def home():

    """

    home()
    Inputs: N/A
    Returns: render_template(index.html)
    Expects: index.html in proj directory
    Notes: N/A
    Purpose: This function presents index.html on running the application

    """

    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def score():

    """

    score()
    Inputs: N/A
    Returns: data (text) and evaluate(data)
    Expects: input in the form in the app
    Notes: N/A
    Purpose: This function calls evaluate() and returns the input text and score in json format
    
    """

    data = request.form['text']
    score = evaluate(data)
    
    # Outputting the result in % form for readability
    score = round(100 * score, 2)

    return jsonify({'Argument': data, 'Quality Score': score})

if __name__ == "__main__":
    app.run(debug=True)