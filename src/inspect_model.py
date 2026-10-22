"""
UNN Community Fake News Detection System
Model inspection - what does the model actually consider Real or Fake?

Part 1: prints the words with the strongest pull toward Real and Fake.
Part 2: runs sample headlines through the model and prints each verdict.

Run from the project root with the virtual environment activated:
    python src/inspect_model.py
"""

import re
import string

import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/logistic_regression.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)


# ---------------------------------------------------------
# PART 1: strongest words in each direction
# ---------------------------------------------------------
words = vectorizer.get_feature_names_out()
weights = model.coef_[0]          # positive = pushes toward Real (label 1)
order = np.argsort(weights)

print("=" * 55)
print("TOP 25 WORDS THE MODEL ASSOCIATES WITH REAL")
print("=" * 55)
for i in order[::-1][:25]:
    print(f"  {words[i]:<20} {weights[i]:+.2f}")

print()
print("=" * 55)
print("TOP 25 WORDS THE MODEL ASSOCIATES WITH FAKE")
print("=" * 55)
for i in order[:25]:
    print(f"  {words[i]:<20} {weights[i]:+.2f}")


# ---------------------------------------------------------
# PART 2: sample headlines
# ---------------------------------------------------------
samples = [
    # formal, wire-service style
    ("WASHINGTON (Reuters) - The Senate voted on Tuesday to approve the budget, officials said.", "formal news"),
    ("The federal government said on Monday it would review the minimum wage, a spokesman told reporters.", "formal news"),
    # genuine-style UNN notices
    ("The Registrar announces that the 2026/2027 post-UTME screening exercise will hold from 10 to 14 August.", "UNN official"),
    ("Management has released the revised academic calendar for the second semester.", "UNN official"),
    # scam / hoax style
    ("BREAKING!!! All students must pay N5,000 immediately or lose admission. Share now!!!", "UNN scam"),
    ("Free laptop for every UNN student! Click this link and send your BVN to claim.", "UNN scam"),
    # calm, well-written fake (the hard case)
    ("Sources confirm that lectures have been suspended indefinitely following a meeting of senate members.", "calm fake"),
    ("The vice chancellor has approved a 50 percent reduction in school fees for all final year students.", "calm fake"),
    # neutral everyday text
    ("The library will now close at 8pm on weekdays.", "neutral campus"),
    ("Students are advised to collect their ID cards from the faculty office.", "neutral campus"),
    # sensational non-UNN
    ("SHOCKING: Celebrity secretly arrested, you won't believe what happened next!", "clickbait"),
    ("Scientists say a new study links regular exercise to lower blood pressure.", "science news"),
]

print()
print("=" * 78)
print("SAMPLE HEADLINES")
print("=" * 78)
print(f"{'Verdict':<8}{'Conf':>7}  {'Type':<16} Text")
print("-" * 78)
for text, kind in samples:
    vec = vectorizer.transform([clean_text(text)])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]
    verdict = "Real" if pred == 1 else "Fake"
    print(f"{verdict:<8}{prob * 100:>6.1f}%  {kind:<16} {text[:48]}...")
