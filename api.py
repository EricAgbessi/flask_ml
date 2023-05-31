import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
import joblib

from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')
stand = joblib.load('scaler.pkl')

@app.after_request
def add_cors_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query = pd.get_dummies(pd.DataFrame(json_, index=[0]))
    query = query.reindex(columns=model_columns, fill_value=0)
    query = stand.transform(query)
    prediction = model.predict(query)

    # Calculate probabilities
    probabilities = model.predict_proba(query)
    positive_probas = probabilities[:, 1]
    negative_probas = probabilities[:, 0]

    return jsonify({
        "prediction": str(prediction[0]),
        "positive_probas": str(positive_probas[0]),
        "negative_probas": str(negative_probas[0])
    })


if __name__ == '__main__':
    app.run(debug=True)