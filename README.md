# UNN Community Fake News Detection System

Final year project — Department of Computer Science, University of Nigeria, Nsukka.

## Project structure

```
unn-fake-news-detector/
├── app.py                 # Flask entry point
├── requirements.txt       # Python dependencies
├── data/
│   ├── raw/                # Original, untouched datasets (ISOT, LIAR, UNN-collected)
│   └── processed/          # Cleaned/preprocessed data ready for training
├── models/                 # Saved trained models (.pkl files)
├── notebooks/               # Jupyter notebooks for experimentation
├── src/                    # Reusable Python modules (preprocessing, training, prediction, SHAP)
├── database/                # SQLite database file + schema.sql
├── templates/               # HTML pages (Flask/Jinja2)
└── static/                  # CSS and JavaScript
```

## Setup instructions (run these on your own machine)

1. **Install Python 3.10+** if you don't have it: https://www.python.org/downloads/

2. **Create a virtual environment** (do this inside the `unn-fake-news-detector` folder):
   ```bash
   python -m venv venv
   ```

3. **Activate it:**
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download NLTK data** (one-time, needed for preprocessing):
   ```bash
   python -m nltk.downloader stopwords punkt wordnet
   ```

6. **Download the spaCy English model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

7. **Run the app to confirm everything works:**
   ```bash
   python app.py
   ```
   Then open http://127.0.0.1:5000 in your browser. You should see a placeholder page.

8. **Initialize Git** (do this once, then commit regularly as you build):
   ```bash
   git init
   git add .
   git commit -m "Initial project scaffold"
   ```

## Project phases

- [x] Phase 1: Environment setup + project scaffold
- [ ] Phase 2: Dataset collection & labelling (ISOT/LIAR + UNN-specific data)
- [ ] Phase 3: Text preprocessing pipeline
- [ ] Phase 4: Feature extraction (TF-IDF) + model training (Logistic Regression, Naive Bayes, SVM)
- [ ] Phase 5: Model evaluation & comparison
- [ ] Phase 6: SHAP explainability integration
- [ ] Phase 7: Flask web interface (submission form + results page)
- [ ] Phase 8: SQLite database (Users, Admin, Submissions, Feedback, Detection tables)
- [ ] Phase 9: Testing & deployment
