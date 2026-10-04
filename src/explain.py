"""
UNN Community Fake News Detection System
Phase 5: SHAP Explainability

This script loads your trained model and shows WHICH WORDS drove a
particular prediction - this is what makes your system explainable
rather than a black box.

Run from the project root with the virtual environment activated:
    python src/explain.py

Note: SHAP works best with models that output probabilities (like
Logistic Regression). If your best model was SVM (LinearSVC), this
script uses Logistic Regression instead for the explanation step,
since LinearSVC doesn't directly support SHAP's KernelExplainer well.
Your main classifier can still be SVM - this is just for explanations.
"""

import joblib
import shap
import numpy as np

# Load the trained vectorizer and the Logistic Regression model specifically
# (SHAP needs probability outputs, which LinearSVC doesn't give directly)
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/logistic_regression.pkl')


def explain_prediction(text, top_n=10):
    """
    Takes a piece of text, predicts Real/Fake, and shows the top words
    that influenced that decision.
    """
    # Clean the text the same way preprocess.py does
    import re
    import string
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(t):
        t = str(t).lower()
        t = re.sub(r'http\S+|www\S+', '', t)
        t = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', t)
        t = re.sub(r'\d+', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        words = t.split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(words)

    cleaned = clean_text(text)
    text_vector = vectorizer.transform([cleaned])

    # Get the prediction and probability
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    label = "Real" if prediction == 1 else "Fake"
    confidence = probability[prediction]

    print(f"\nPrediction: {label} (confidence: {confidence:.2%})")

    # Use SHAP's LinearExplainer, which is fast and built specifically
    # for linear models like Logistic Regression
    explainer = shap.LinearExplainer(model, vectorizer.transform(
        ["dummy text for background"]  # small background sample for speed
    ))
    shap_values = explainer.shap_values(text_vector)

    # Get feature names (the actual words) and their SHAP importance scores
    feature_names = vectorizer.get_feature_names_out()
    text_array = text_vector.toarray()[0]
    shap_array = shap_values[0]

    # Only look at words that actually appear in this text
    word_importance = []
    for i, value in enumerate(text_array):
        if value > 0:  # word is present in the text
            word_importance.append((feature_names[i], shap_array[i]))

    # Sort by absolute importance (most influential words first)
    word_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop {top_n} words influencing this prediction:")
    print(f"{'Word':<20} {'Impact':>10}  {'Direction'}")
    print("-" * 45)
    for word, score in word_importance[:top_n]:
        direction = "→ Real" if score > 0 else "→ Fake"
        print(f"{word:<20} {score:>10.4f}  {direction}")

    return {
        'label': label,
        'confidence': confidence,
        'word_importance': word_importance[:top_n]
    }


if __name__ == '__main__':
    # Example usage - try it on a sample piece of text
    sample_text = input("Paste a piece of news text to analyze: ")
    explain_prediction(sample_text)
