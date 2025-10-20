# Quantitative Stock Return Prediction Model

A simple **quantitative finance project** that predicts next-day stock returns using linear regression on historical data.  
Perfect for students, finance enthusiasts, or anyone learning quant research.

---

## 📈 Features
- Downloads stock data from Yahoo Finance (via `yfinance`)
- Generates lag, volatility, and moving average features
- Trains a linear regression model
- Evaluates accuracy (R², MSE)
- Exports predictions to `model_predictions.csv`

---

## 🧠 How It Works
1. Pull historical price data.
2. Create lag features and technical indicators.
3. Train a model to predict next-day returns.
4. Evaluate and export results.

---

## 🧰 Installation

```bash
git clone https://github.com/alexxsssun-byte/quant
cd quant-stock-model
pip install -r requirements.txt
