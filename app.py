import os
from flask import Flask, request, jsonify
import pmdarima as pm
import numpy as np
import pandas as pd
import logging
import threading # Untuk menjalankan Flask di thread terpisah

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_arima():
    if not request.is_json:
        logger.error("Request must be JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    historical_prices = data.get('data')
    forecast_horizon = data.get('forecast_horizon')

    if not historical_prices or not isinstance(historical_prices, list):
        logger.error("Missing or invalid 'data' (historical_prices) in request")
        return jsonify({"error": "Missing or invalid 'data' (historical_prices)"}), 400

    if not forecast_horizon or not isinstance(forecast_horizon, int) or forecast_horizon <= 0:
        logger.error("Missing or invalid 'forecast_horizon' in request. Must be a positive integer.")
        return jsonify({"error": "Missing or invalid 'forecast_horizon'. Must be a positive integer."}), 400

    try:
        # Konversi list harga ke Pandas Series
        series = pd.Series(historical_prices)

        logger.info(f"Received {len(historical_prices)} historical data points.")
        logger.info(f"Forecast horizon: {forecast_horizon} days.")

        # Menggunakan auto_arima untuk secara otomatis menemukan parameter (p,d,q) terbaik
        model = pm.auto_arima(series,
                              seasonal=False,
                              stepwise=True,
                              suppress_warnings=True,
                              error_action="ignore",
                              max_order=None,
                              trace=False
                             )

        logger.info(f"ARIMA model fitted with order: {model.order}")

        # Lakukan prediksi
        predictions = model.predict(n_periods=forecast_horizon).tolist()

        logger.info(f"Generated {len(predictions)} predictions.")

        return jsonify({"predictions": predictions}), 200

    except Exception as e:
        logger.error(f"Error during ARIMA prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Fungsi untuk menjalankan Flask app di thread terpisah
def run_flask_app():
    # Colab secara default akan menjalankan di localhost, port 5001
    app.run(host='0.0.0.0', port=5001)
