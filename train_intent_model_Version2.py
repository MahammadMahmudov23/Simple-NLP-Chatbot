import json
import pickle
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Prepare data
texts = []
labels = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['tag'])

# Vectorize text
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(texts)

# Train classifier
clf = MultinomialNB()
clf.fit(X, labels)

# Save model and vectorizer
with open('intent_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, clf), f)

print("Training complete. Model saved as intent_model.pkl")