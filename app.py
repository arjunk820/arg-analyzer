from flask import Flask, request, jsonify
from model import score

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def score():
    data = request.json
    text = data['text']
    result = score(text)
    return jsonify({'quality_score': result})

@app.route("/")
def home():
    return "Welcome to the LawyerAI!"

if __name__ == "__main__":
    app.run(debug=True)