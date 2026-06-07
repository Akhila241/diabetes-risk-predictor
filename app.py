from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load models
with open('model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load accuracies and feature importance
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

@app.route('/')
def home():
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [
        float(data['pregnancies']),
        float(data['glucose']),
        float(data['bloodpressure']),
        float(data['skinthickness']),
        float(data['insulin']),
        float(data['bmi']),
        float(data['diabetespedigree']),
        float(data['age'])
    ]

    input_array = np.array([features])

    # Logistic Regression prediction
    lr_pred = lr_model.predict(input_array)[0]
    lr_prob = round(float(lr_model.predict_proba(input_array)[0][1]) * 100, 1)

    # Random Forest prediction
    rf_pred = rf_model.predict(input_array)[0]
    rf_prob = round(float(rf_model.predict_proba(input_array)[0][1]) * 100, 1)

    result = {
        'prediction': int(lr_pred),
        'probability': lr_prob,
        'message': 'High risk of diabetes' if lr_pred == 1 else 'Low risk of diabetes',
        'rf_prediction': int(rf_pred),
        'rf_probability': rf_prob,
        'rf_message': 'High risk of diabetes' if rf_pred == 1 else 'Low risk of diabetes',
        'feature_importance': model_info['feature_importance']
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)