"""
UNN Community Fake News Detection System
Flask web application - Phases 6-8 (web interface, database, accounts, admin)

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
import os
import re
import sqlite3
import string

import joblib
import shap
from flask import Flask, render_template, request, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Admin login settings. For a real deployment, set these as environment
# variables instead of relying on the defaults below.
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-secret-key')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'unnadmin2026')

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
           VALUES (?, ?, ?, ?, ?)""",
        (session.get('user_id'), submitted_text, category, result['label'], result['confidence'])
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


@app.context_processor
def inject_user():
    """Makes the logged-in user's name available to every template."""
    return {'current_user_name': session.get('user_name')}


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')

        if not full_name or not email or not password:
            error = "Please fill in every field."
        elif '@' not in email:
            error = "Please enter a valid email address."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "The two passwords do not match."
        else:
            conn = get_db()
            exists = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
            if exists:
                conn.close()
                error = "An account with that email already exists."
            else:
                cursor = conn.execute(
                    "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
                    (full_name, email, generate_password_hash(password))
                )
                conn.commit()
                session['user_id'] = cursor.lastrowid
                session['user_name'] = full_name
                conn.close()
                return redirect('/')
    return render_template('register.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['user_id']
            session['user_name'] = user['full_name']
            return redirect('/')
        error = "Incorrect email or password."
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    return redirect('/')


@app.route('/history')
def history():
    if not session.get('user_id'):
        return redirect('/login')

    conn = get_db()
    rows = conn.execute(
        """SELECT submitted_text, category, verdict, confidence_score, submitted_at
           FROM submissions WHERE user_id = ? ORDER BY submission_id DESC""",
        (session['user_id'],)
    ).fetchall()
    conn.close()
    return render_template('history.html', rows=rows)


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect('/admin')
        error = "Incorrect password."
    return render_template('admin_login.html', error=error)


@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect('/')


@app.route('/admin')
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect('/admin/login')

    conn = get_db()

    total = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
    users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    fake = conn.execute("SELECT COUNT(*) FROM submissions WHERE verdict = 'Fake'").fetchone()[0]
    real = total - fake

    agree = conn.execute("SELECT COUNT(*) FROM feedback WHERE user_response = 'Agree'").fetchone()[0]
    disagree = conn.execute("SELECT COUNT(*) FROM feedback WHERE user_response = 'Disagree'").fetchone()[0]
    feedback_total = agree + disagree

    by_category = conn.execute(
        """SELECT category,
                  COUNT(*) AS total,
                  SUM(CASE WHEN verdict = 'Fake' THEN 1 ELSE 0 END) AS fake_count
           FROM submissions GROUP BY category ORDER BY total DESC"""
    ).fetchall()

    recent = conn.execute(
        """SELECT s.submission_id, s.submitted_text, s.category, s.verdict,
                  s.confidence_score, s.submitted_at,
                  (SELECT user_response FROM feedback f
                   WHERE f.submission_id = s.submission_id
                   ORDER BY f.feedback_id DESC LIMIT 1) AS feedback
           FROM submissions s ORDER BY s.submission_id DESC LIMIT 25"""
    ).fetchall()
    conn.close()

    stats = {
        'total': total,
        'users': users,
        'fake': fake,
        'real': real,
        'fake_pct': round(fake / total * 100, 1) if total else 0,
        'agree': agree,
        'disagree': disagree,
        'feedback_total': feedback_total,
        'agree_pct': round(agree / feedback_total * 100, 1) if feedback_total else 0,
    }
    return render_template('admin.html', stats=stats, by_category=by_category, recent=recent)


if __name__ == '__main__':
    app.run(debug=True)
