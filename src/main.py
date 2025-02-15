from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
from Preprocess.preprocess import Preprocess
import joblib


app = Flask(__name__)

def load_model():
    model_path = os.path.join(os.path.join(os.path.dirname(__file__), "Model", "models"))
    ensemble_model = joblib.load(os.path.join(model_path, "ensemble_model.pkl"))
    return ensemble_model

@app.route("/")
def home():
    return "<h2> Bank Customer Churn Prediction</h2>"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}") 

        if not data:
            return jsonify({"error": "No input data found"}), 400
        
        dataframe = pd.DataFrame(data)
        dataframe.reset_index(drop=True, inplace=True)

        preprocess = Preprocess(dataframe)
        preprocess_data = preprocess.preprocess_pipeline()
        model = load_model()
        predict = model.predict(preprocess_data)

        return jsonify({"Predict": predict.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")
