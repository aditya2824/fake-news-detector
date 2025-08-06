from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import string

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Prediction counters
prediction_counts = {"real": 0, "fake": 0}

# Text cleaning function
def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction_text=None,
                           confidence_score=None,
                           real_count=prediction_counts["real"],
                           fake_count=prediction_counts["fake"])

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        clean_text = preprocess(message)
        vect = vectorizer.transform([clean_text]).toarray()
        prediction = model.predict(vect)
        prob = model.predict_proba(vect)[0]
        confidence = round(max(prob) * 100, 2)

        if prediction[0] == 1:
            label = "🟢 This is likely Real News ✅"
            prediction_counts["real"] += 1
        else:
            label = "🔴 This appears to be Fake News ❌"
            prediction_counts["fake"] += 1

        return render_template("index.html",
                               prediction_text=label,
                               confidence_score=confidence,
                               real_count=prediction_counts["real"],
                               fake_count=prediction_counts["fake"])

if __name__ == "__main__":
    app.run(debug=True)
