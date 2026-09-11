"""
UNN Community Fake News Detection System
Entry point for the Flask web application.

Right now this is a placeholder route just to confirm your environment
is set up correctly. We will build out the real submission/prediction
routes once the ML model is trained (Phase 2 of the project).
"""

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
