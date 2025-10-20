#!/usr/bin/env python3
"""
Streamlit App for Stock Return Prediction Model
Author: Your Name
GitHub: https://github.com/yourusername/quant-stock-model
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stock_return_model import load_data, prepare_features, split_data, train_and_evaluate

st.set_page_config(page_title="Quant Stock Return Model", page_icon="📈", layout="centered")

# --- Sidebar ---
st.sidebar.title("⚙️ Model Settings")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Model 🚀")

# --- Title ---
st.title("📊 Quantitative Stock Return Prediction")
st.markdown("Predict next-day stock returns using a simple **linear regression model** based on historical data.")

# --- Run Model ---
if run_button:
    with st.spinner("Building model... please wait"):
        data = load_data(ticker, start=start_date, end=end_date)
        data = prepare_features(data)
        X_train, X_test, y_train, y_test = split_data(data)
        r2, mse, results = train_and_evaluate(X_train, y_train, X_test, y_test)

        st.success("✅ Model completed successfully!")

        # --- Metrics ---
        st.subheader("📈 Model Performance")
        st.metric(label="R² Score", value=f"{r2:.4f}")
        st.metric(label="Mean Squared Error", value=f"{mse:.6f}")

        # --- Plot actual vs predicted ---
        st.subheader("🔍 Predicted vs Actual Returns")
        plt.figure(figsize=(8, 4))
        plt.plot(results["Actual"].values, label="Actual", alpha=0.7)
        plt.plot(results["Predicted"].values, label="Predicted", alpha=0.7)
        plt.legend()
        plt.title(f"{ticker} Return Prediction")
        plt.xlabel("Test Period")
        plt.ylabel("Return")
        st.pyplot(plt)

        # --- Show table ---
        st.subheader("🧮 Sample Predictions")
        st.dataframe(results.head(10))

        # --- Download results ---
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions CSV", data=csv, file_name=f"{ticker}_predictions.csv")

else:
    st.info("👈 Set your parameters on the left and click **Run Model 🚀** to begin.")
