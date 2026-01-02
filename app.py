from flask import Flask, request, jsonify
import pandas as pd # Import pandas for DataFrame creation

import joblib

app = Flask(__name__)

# Load the trained model
model_filename = 'logistic_regression_model.joblib'
log_reg_model = joblib.load(model_filename)

print(f"Logistic Regression model successfully loaded from '{model_filename}'.")

# Initialize Flask application (already done in previous cell, but re-included for completeness)


@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json(force=True)

    # Extract features from the received data
    # The order of features should match the training data
    # Features: 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    try:
        features = [
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]
    except KeyError as e:
        return jsonify({'error': f'Missing feature in request: {e}'}), 400

    # Convert features to a pandas DataFrame, matching the model's expected input format
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame([features], columns=feature_names)

    # Make prediction using the loaded model
    prediction_array = log_reg_model.predict(input_df)

    # Convert the prediction (numpy array) to a standard Python integer
    prediction = int(prediction_array[0])

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

print("Flask application 'app' and '/predict' endpoint updated with prediction logic.")

if __name__ == "__main__":
  app.run(debug=True)
