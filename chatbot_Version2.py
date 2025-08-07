"""
chatbot.py

Command-line chatbot using a trained intent recognition model and simple NLP.
"""

import json
import pickle
import random
import nltk

nltk.download('punkt')

# Load intents and trained model
with open('intents.json', 'r') as f:
    intents = json.load(f)
with open('intent_model.pkl', 'rb') as f:
    vectorizer, clf = pickle.load(f)

def get_intent(text):
    """
    Predict the intent of the user's input.
    """
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]
    max_prob = max(probs)
    if max_prob < 0.4:
        return None
    return clf.classes_[probs.argmax()]

def get_response(intent_tag):
    """
    Return a random response from the matched intent.
    """
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

def chat():
    """
    Start the command-line chat interface.
    """
    print("Start chatting (type 'quit' to exit):")
    while True:
        inp = input("> ")
        if inp.lower() == "quit":
            print("Goodbye!")
            break
        intent = get_intent(inp)
        if intent:
            print(get_response(intent))
        else:
            print("Sorry, I didn't understand that.")

if __name__ == "__main__":
    chat()