# app.py
from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
from joblib import load
from utils.feature_names import FEATURE_NAMES
import os

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-key"

MODEL_PATH = "model.joblib"

# Load model
if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model not found. Please run train_model.py to create {MODEL_PATH}")

model = load(MODEL_PATH)  # pipeline: scaler + model

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", feature_names=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read inputs in same order as FEATURE_NAMES
        values = []
        for name in FEATURE_NAMES:
            val = request.form.get(name, "").strip()
            if val == "":
                flash(f"Missing value for {name}. Please enter all {len(FEATURE_NAMES)} features.", "danger")
                return redirect(url_for("index"))
            # convert safely
            try:
                fv = float(val)
            except ValueError:
                flash(f"Invalid number for {name}: {val}", "danger")
                return redirect(url_for("index"))
            values.append(fv)

        X = np.array(values, dtype=np.float32).reshape(1, -1)
        pred = model.predict(X)[0]        # 1 or 0
        prob = model.predict_proba(X)[0].max()

        label = "cancerous" if int(pred) == 1 else "not cancerous"
        return render_template("index.html", feature_names=FEATURE_NAMES, result=label, prob=prob)
    except Exception as e:
        flash(f"Error during prediction: {e}", "danger")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
