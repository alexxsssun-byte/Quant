#!/usr/bin/env python3
"""
Stock Return Prediction Model
Author: Your Name
GitHub: https://github.com/yourusername/quant-stock-model
Description:
  A quantitative model to predict next-day stock returns using
  linear regression on historical financial data.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Optional: use real stock data if yfinance is available
try:
    import yfinance as yf
    REAL_DATA = True
except ImportError:
    REAL_DATA = False


def load_data(ticker="AAPL", start="2015-01-01", end="2025-01-01"):
    """Load real or simulated stock data."""
    if REAL_DATA:
        print(f"📥 Downloading {ticker} data from Yahoo Finance...")
        data = yf.download(ticker, start=start, end=end)
    else:
        print("⚙️ Simulating synthetic stock data (no yfinance installed).")
        np.random.seed(42)
        dates = pd.date_range(start=start, end=end, freq="B")
        price = 150 + np.cumsum(np.random.normal(0, 1, len(dates)))
        data = pd.DataFrame({"Date": dates, "Close": price}).set_index("Date")
    return data


def prepare_features(data):
    """Generate lag, moving average, and volatility features."""
    data['Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Lag1'] = data['Return'].shift(1)
    data['Lag2'] = data['Return'].shift(2)
    data['MA5'] = data['Close'].rolling(5).mean() / data['Close'] - 1
    data['Vol10'] = data['Return'].rolling(10).std()
    data = data.dropna()
    return data


def split_data(data):
    """Train-test split."""
    train = data.loc[:'2023-12-31']
    test = data.loc['2024-01-01':]

    X_train = train[['Lag1', 'Lag2', 'MA5', 'Vol10']]
    y_train = train['Return'].shift(-1).dropna()
    X_train = X_train.iloc[:-1, :]

    X_test = test[['Lag1', 'Lag2', 'MA5', 'Vol10']]
    y_test = test['Return'].shift(-1).dropna()
    X_test = X_test.iloc[:-1, :]

    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train model and evaluate performance."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    results = pd.DataFrame({"Actual": y_test, "Predicted": preds}).reset_index(drop=True)
    return r2, mse, results


def main():
    parser = argparse.ArgumentParser(description="Quantitative Stock Return Model")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    args = parser.parse_args()

    print("\n🚀 Building stock return prediction model...\n")
    data = load_data(args.ticker)
    data = prepare_features(data)
    X_train, X_test, y_train, y_test = split_data(data)
    r2, mse, results = train_and_evaluate(X_train, y_train, X_test, y_test)

    print("\n📊 MODEL PERFORMANCE")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    print("\nSample predictions:")
    print(results.head())

    results.to_csv("model_predictions.csv", index=False)
    print("\n✅ Results saved to 'model_predictions.csv'")


if __name__ == "__main__":
    main()
