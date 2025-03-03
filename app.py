#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, reqparse
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
api = Api(app, version="1.0", title="Prediction API", description="API for predicting weight based on calorie intake")

ns = api.namespace("predict", description="Prediction operations")

# Healthcheck Route for Railway
@app.route("/")
def healthcheck():
    return jsonify({"message": "API is running"}), 200

# Load the trained models
try:
    lstm_model = load_model("lstm_model.h5")
    linear_regressor = joblib.load("linear_regressor.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    lstm_model, linear_regressor, scaler = None, None, None

upload_parser = reqparse.RequestParser()
upload_parser.add_argument("file", type=FileStorage, location="files", required=True, help="CSV file")

@ns.route("/")
class Predict(Resource):
    @api.expect(upload_parser)
    def post(self):
        if lstm_model is None or linear_regressor is None or scaler is None:
            return {"error": "Model is not available"}, 500
        
        args = upload_parser.parse_args()
        file = args["file"]

        if not file.filename.endswith(".csv"):
            return {"error": "Only CSV files are allowed"}, 400

        try:
            data = pd.read_csv(file)
            if {"Weight", "Calories intake"}.issubset(data.columns):
                scaled = scaler.transform(data[["Weight", "Calories intake"]]).reshape(1, 7, 2)
                lstm_output = lstm_model.predict(scaled)
                flat = lstm_output.reshape(1, -1)
                prediction = linear_regressor.predict(flat)
                return {"predicted_weight": prediction.tolist()}, 200
            else:
                return {"error": "Invalid CSV Columns"}, 400
        except Exception as e:
            return {"error": str(e)}, 500

api.add_namespace(ns, path="/predict")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
