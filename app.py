from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "logistic_regression_model.joblib"

# Load model
log_reg_model = joblib.load(MODEL_PATH)
print("Model loaded successfully")

@app.route("/")
def home():
    return jsonify({"status": "ML API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

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
    except KeyError as e:
        return jsonify({"error": f"Missing field {e}"}), 400

    prediction = int(log_reg_model.predict(input_df)[0])
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
