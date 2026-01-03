from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "logistic_regression_model.joblib"

try:
    log_reg_model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model: {e}")
    log_reg_model = None


@app.route("/predict", methods=["POST"])
def predict():
    if log_reg_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(force=True)

    required_features = [
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
        input_df = pd.DataFrame(
            [[data[feature] for feature in required_features]],
            columns=required_features
        )
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400

    prediction = int(log_reg_model.predict(input_df)[0])

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
