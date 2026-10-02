"""
UNN Community Fake News Detection System
Phase 4: Model Training

This script:
1. Loads your cleaned combined dataset
2. Converts text into numbers using TF-IDF
3. Splits data into training (80%) and testing (20%) sets
4. Trains three models: Logistic Regression, Naive Bayes, SVM
5. Evaluates each one and prints accuracy, precision, recall, F1
6. Saves the best model + the TF-IDF vectorizer for later use in Flask

Run from the project root with the virtual environment activated:
    python src/train_models.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# =========================================================
# STEP 1: LOAD THE CLEANED DATA
# =========================================================

print("Loading cleaned dataset...")
df = pd.read_csv('data/processed/combined_clean.csv')
df = df.dropna(subset=['clean_text'])
print(f"Loaded {len(df)} rows")

X = df['clean_text']
y = df['label']


# =========================================================
# STEP 2: SPLIT INTO TRAIN/TEST SETS
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} rows")
print(f"Test set: {len(X_test)} rows")


# =========================================================
# STEP 3: CONVERT TEXT TO TF-IDF FEATURES
# =========================================================

print("\nConverting text to TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =========================================================
# STEP 4: TRAIN AND EVALUATE EACH MODEL
# =========================================================

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(max_iter=2000)
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print('='*50)

    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}")

    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    trained_models[name] = model


# =========================================================
# STEP 5: PICK THE BEST MODEL AND SAVE EVERYTHING
# =========================================================

best_name = max(results, key=lambda n: results[n]['accuracy'])
best_model = trained_models[best_name]

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_name} (Accuracy: {results[best_name]['accuracy']:.4f})")
print('='*50)

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

# Also save all three models individually, in case you want to compare/discuss all of them
for name, model in trained_models.items():
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(model, f'models/{safe_name}.pkl')

print("\nSaved best_model.pkl and tfidf_vectorizer.pkl to models/")
print("Saved all three individual models to models/ as well")
print("\nSummary of all results:")
for name, metrics in results.items():
    print(f"  {name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
