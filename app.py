from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model
model = joblib.load("logistic_regression_model.joblib")

# -----------------------------
# UI route
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# ML prediction route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    try:
        input_df = pd.DataFrame([[data[f] for f in features]], columns=features)
        prediction = int(model.predict(input_df)[0])
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------
# Run (Render compatible)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
