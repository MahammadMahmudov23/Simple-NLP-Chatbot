import json
import pickle
import random
import nltk

nltk.download('punkt')

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

def chat():
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