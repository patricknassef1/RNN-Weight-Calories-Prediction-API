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

# Load the trained models
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
upload_parser.add_argument("file", type=FileStorage, location="files", required=True, help="CSV file")

@ns.route("/")
class Predict(Resource):
    @api.expect(upload_parser)
    @api.response(200, "Success")
    @api.response(400, "Invalid input file")
    @api.response(500, "Internal server error")
    def post(self):
        """Predict weight based on past 7 days of weight and calorie intake"""
        if lstm_model is None or linear_regressor is None or scaler is None:
            return {"error": "Model is not available"}, 500
        
        try:
            args = upload_parser.parse_args()
            file = args.get("file")

            if not file or not isinstance(file, FileStorage):
                return {"error": "Invalid file upload"}, 400

            if not file.filename.endswith(".csv"):
                return {"error": "Only CSV files are allowed"}, 400

            input_data = pd.read_csv(file)

            # Ensure required columns exist
            required_columns = {"Weight", "Calories intake"}
            if not required_columns.issubset(input_data.columns):
                return {"error": f"CSV file must contain columns: {required_columns}"}, 400

            # Preprocess input data
            inputs_scaled = scaler.transform(input_data[['Weight', 'Calories intake']])
            inputs_scaled = inputs_scaled.reshape(1, 7, 2)  # Reshape for LSTM input

            # Get LSTM predictions
            lstm_output = lstm_model.predict(inputs_scaled)

            # Flatten LSTM output for linear regression
            lstm_output_flat = lstm_output.reshape(1, -1)
            predicted_weight = linear_regressor.predict(lstm_output_flat)

            # Reshape for inverse transformation
            predicted_weight_reshaped = np.reshape(predicted_weight, (-1, 1))
            predicted_values_with_zeros = np.hstack((predicted_weight_reshaped, np.zeros_like(predicted_weight_reshaped)))
            predicted_weight_final = scaler.inverse_transform(predicted_values_with_zeros)[:, 0]

            return {"predicted_weight": predicted_weight_final.tolist()}, 200

        except Exception as e:
            return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
