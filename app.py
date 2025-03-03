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
import os

app = Flask(__name__)
api = Api(app, version="1.0", title="Prediction API", description="API for predicting weight based on calorie intake")
ns = api.namespace("predict", description="Prediction operations")

# Load models
try:
    lstm_model = load_model("lstm_model.h5")
    linear_regressor = joblib.load("linear_regressor.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    lstm_model, linear_regressor, scaler = None, None, None

# File upload parser
upload_parser = reqparse.RequestParser()
upload_parser.add_argument("file", type=FileStorage, location="files", required=True, help="Upload a CSV file")

@ns.route("/")
class Predict(Resource):
    @api.expect(upload_parser)
    @api.response(200, "Success")
    @api.response(400, "Invalid input file")
    @api.response(500, "Internal server error")
    def post(self):
        if not lstm_model or not linear_regressor or not scaler:
            return {"error": "Models not loaded"}, 500

        args = upload_parser.parse_args()
        file = args.get("file")
        if not file or not file.filename.endswith(".csv"):
            return {"error": "Invalid file. Please upload a CSV file"}, 400

        try:
            data = pd.read_csv(file)
            required_columns = {"Weight", "Calories intake"}
            if not required_columns.issubset(data.columns):
                return {"error": f"CSV must contain columns: {required_columns}"}, 400

            inputs_scaled = scaler.transform(data[['Weight', 'Calories intake']])
            inputs_scaled = inputs_scaled.reshape(1, 7, 2)
            lstm_output = lstm_model.predict(inputs_scaled)
            lstm_output_flat = lstm_output.reshape(1, -1)
            predicted_weight = linear_regressor.predict(lstm_output_flat)
            predicted_weight_reshaped = np.reshape(predicted_weight, (-1, 1))
            predicted_values_with_zeros = np.hstack((predicted_weight_reshaped, np.zeros_like(predicted_weight_reshaped)))
            predicted_weight_final = scaler.inverse_transform(predicted_values_with_zeros)[:, 0]

            return jsonify({"predicted_weight": predicted_weight_final.tolist()})
        except Exception as e:
            return {"error": str(e)}, 500

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Prediction API"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Railway PORT or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
