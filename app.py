"""
UNN Community Fake News Detection System
Flask web application - Phases 6-7 (web interface + database)

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

import json
import re
import sqlite3
import string

import joblib
import shap
from flask import Flask, render_template, request, redirect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

DB_PATH = 'database/unn_fnd.db'


def get_db():
    """Opens a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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
    category = request.form.get('category', 'General')

    if not submitted_text:
        return render_template('index.html', error="Please paste some text to analyze.")

    result = analyze_text(submitted_text)

    # Save this submission to the database
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO submissions (user_id, submitted_text, category, verdict, confidence_score)
           VALUES (NULL, ?, ?, ?, ?)""",
        (submitted_text, category, result['label'], result['confidence'])
    )
    submission_id = cursor.lastrowid

    # If it was flagged Fake, also log the SHAP explanation
    if result['label'] == 'Fake':
        cursor.execute(
            """INSERT INTO fake_news_detection (submission_id, fake_news_type, shap_explanation)
               VALUES (?, ?, ?)""",
            (submission_id, 'Unclassified', json.dumps(result['top_words']))
        )

    conn.commit()
    conn.close()

    result['submission_id'] = submission_id
    return render_template('results.html', result=result)


@app.route('/feedback', methods=['POST'])
def feedback():
    submission_id = request.form.get('submission_id')
    user_response = request.form.get('response')  # 'Agree' or 'Disagree'

    conn = get_db()
    conn.execute(
        "INSERT INTO feedback (submission_id, user_response) VALUES (?, ?)",
        (submission_id, user_response)
    )
    conn.commit()
    conn.close()

    return render_template('feedback_thanks.html')


if __name__ == '__main__':
    app.run(debug=True)
