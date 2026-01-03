from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib, pandas as pd, os

app = Flask(__name__)
CORS(app)

model = joblib.load("logistic_regression_model.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    cols = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age"
    ]
    df = pd.DataFrame([[data[c] for c in cols]], columns=cols)
    return jsonify({"prediction": int(model.predict(df)[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
