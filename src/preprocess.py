"""
UNN Community Fake News Detection System
Phase 3: Data Preprocessing

This script does two things:
1. Combines the three datasets (ISOT, LIAR, UNN) into one unified table
   with just two columns: text, label (where label is 0=Fake, 1=Real)
2. Cleans the text (lowercase, remove noise, remove stopwords, lemmatize)

Run this from the project root folder with the virtual environment activated:
    python src/preprocess.py

Output: data/processed/combined_clean.csv
"""

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- One-time NLTK setup (safe to run every time, skips if already downloaded) ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# =========================================================
# STEP 1: LOAD EACH DATASET AND STANDARDIZE TO text + label
# =========================================================

def load_isot():
    """ISOT has two separate files: Fake.csv and True.csv"""
    fake = pd.read_csv('data/raw/Fake.csv')
    real = pd.read_csv('data/raw/True.csv')

    fake['label'] = 0  # 0 = Fake
    real['label'] = 1  # 1 = Real

    # ISOT has 'title' and 'text' columns separately - combine them
    fake['text'] = fake['title'].fillna('') + ' ' + fake['text'].fillna('')
    real['text'] = real['title'].fillna('') + ' ' + real['text'].fillna('')

    isot = pd.concat([fake[['text', 'label']], real[['text', 'label']]], ignore_index=True)
    print(f"Loaded ISOT: {len(isot)} rows")
    return isot


def load_liar():
    """LIAR is tab-separated, no header row, 14 columns.
    Column 2 (index 1) is the statement text, column 3 (index 2) is the truth label.
    We simplify LIAR's 6-way label into binary Real/Fake."""
    cols = ['id', 'label_raw', 'statement', 'subject', 'speaker', 'job', 'state',
            'party', 'barely_true_c', 'false_c', 'half_true_c', 'mostly_true_c',
            'pants_fire_c', 'context']
    liar = pd.read_csv('data/raw/train.tsv', sep='\t', header=None, names=cols)

    # Map LIAR's 6-way labels to binary: treat clearly false statements as Fake,
    # everything else as Real (this simplification is common in the literature)
    fake_labels = ['false', 'pants-fire', 'barely-true']
    liar['label'] = liar['label_raw'].apply(lambda x: 0 if x in fake_labels else 1)
    liar = liar.rename(columns={'statement': 'text'})

    liar = liar[['text', 'label']].dropna(subset=['text'])
    print(f"Loaded LIAR: {len(liar)} rows")
    return liar


def load_unn():
    """Your UNN-specific dataset - already has text and label columns,
    just need to convert Real/Fake text into 1/0."""
    unn = pd.read_csv('data/raw/unn_dataset.csv')
    unn['label'] = unn['label'].apply(lambda x: 1 if str(x).strip().lower() == 'real' else 0)
    unn = unn[['text', 'label']].dropna(subset=['text'])
    print(f"Loaded UNN dataset: {len(unn)} rows")
    return unn


# =========================================================
# STEP 2: CLEAN THE TEXT
# =========================================================

def clean_text(text):
    """Lowercase, strip URLs/punctuation/numbers, remove stopwords, lemmatize."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)                    # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()            # collapse extra whitespace

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':
    print("Loading datasets...")
    isot = load_isot()
    liar = load_liar()
    unn = load_unn()

    print("\nCombining all datasets...")
    combined = pd.concat([isot, liar, unn], ignore_index=True)
    print(f"Total combined rows: {len(combined)}")

    print("\nCleaning text (this may take a minute for large datasets)...")
    combined['clean_text'] = combined['text'].apply(clean_text)

    # Drop rows that became empty after cleaning
    combined = combined[combined['clean_text'].str.len() > 0]

    print(f"\nFinal dataset size: {len(combined)} rows")
    print(f"Real (1): {(combined['label'] == 1).sum()}")
    print(f"Fake (0): {(combined['label'] == 0).sum()}")

    combined[['clean_text', 'label']].to_csv('data/processed/combined_clean.csv', index=False)
    print("\nSaved to data/processed/combined_clean.csv")
