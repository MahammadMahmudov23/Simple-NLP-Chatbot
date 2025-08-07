from flask import Flask, render_template, request, jsonify
import json
import pickle
import random
import nltk

nltk.download('punkt')

app = Flask(__name__)

# Load intents and model
with open('intents.json', 'r') as f:
    intents = json.load(f)
with open('intent_model.pkl', 'rb') as f:
    vectorizer, clf = pickle.load(f)

def get_intent(text):
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]
    max_prob = max(probs)
    if max_prob < 0.4:
        return None
    return clf.classes_[probs.argmax()]

def get_response(intent_tag):
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    intent = get_intent(user_message)
    if intent:
        response = get_response(intent)
    else:
        response = "Sorry, I didn't understand that."
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)