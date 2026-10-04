"""
UNN Community Fake News Detection System
Flask web application - Phase 6

Routes:
  GET  /          - the submission form
  POST /analyze   - processes submitted text, runs the model + SHAP,
                     and shows the results page

Note on which model is "live" here: your SVM scored highest on raw
accuracy, but SVM (LinearSVC) doesn't output probabilities or play
well with SHAP's explainer. Since showing a confidence score AND an
explanation is central to your project, this app uses the Logistic
Regression model instead - it scored almost identically (91.00% vs
91.28%), and supports both features properly. This tradeoff is worth
mentioning in your report: slightly lower raw accuracy, in exchange
for genuine explainability.
"""

import re
import string

import joblib
import shap
from flask import Flask, render_template, request, redirect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the trained vectorizer and model once, when the app starts
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/logistic_regression.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Same cleaning logic used during preprocessing/training -
    lowercase, strip URLs/punctuation/numbers, remove stopwords, lemmatize."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)


def analyze_text(raw_text):
    """Cleans the text, predicts Real/Fake, and returns the top
    SHAP-explained words behind that prediction."""
    cleaned = clean_text(raw_text)
    text_vector = vectorizer.transform([cleaned])

    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    label = "Real" if prediction == 1 else "Fake"
    confidence = probability[prediction]

    explainer = shap.LinearExplainer(model, vectorizer.transform(["dummy background text"]))
    shap_values = explainer.shap_values(text_vector)

    feature_names = vectorizer.get_feature_names_out()
    text_array = text_vector.toarray()[0]
    shap_array = shap_values[0]

    word_importance = []
    for i, value in enumerate(text_array):
        if value > 0:
            word_importance.append((feature_names[i], float(shap_array[i])))
    word_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        'label': label,
        'confidence': round(confidence * 100, 1),
        'top_words': word_importance[:10],
        'original_text': raw_text
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        # Someone refreshed the results page or visited /analyze directly -
        # send them back to the form instead of showing an error
        return redirect('/')

    submitted_text = request.form.get('news_text', '').strip()

    if not submitted_text:
        return render_template('index.html', error="Please paste some text to analyze.")

    result = analyze_text(submitted_text)
    return render_template('results.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)