"""
stock_return_model.py
Advanced version — integrates technicals, fundamentals, news sentiment, and macro data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Optional libraries (loaded only if used)
from fmp_python.fmp import FMP
from fredapi import Fred
from transformers import pipeline
import torch


# ------------------ CONFIG ------------------
FMP_API_KEY = "soFZaj9OS0tmLv0UzAk2YACP1bIl4fie"
FRED_API_KEY = "5c86a917dbe0247c98b5e9f148833004"

# Initialize FinBERT (for news sentiment)
_sentiment_pipe = None
def get_finbert():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    return _sentiment_pipe


# ------------------ LOAD DATA ------------------
import os
import requests
import pandas as pd

def load_data(ticker, start, end):
    """
    Auto-detects if asset is stock or crypto and loads OHLCV data from Twelve Data.
    Stocks: 'SPY', 'MSFT'
    Crypto: 'BTC/USD', 'ETH/USD'
    """
    api_key = os.getenv("TWELVEDATA_API_KEY", "2d97ca468f30408681b689dd92437c7a")

    # --- Determine market type ---
    is_crypto = "/" in ticker

    # --- Choose interval ---
    interval = "1day"

    # --- Build API URL ---
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={ticker}&interval={interval}&apikey={api_key}&outputsize=5000"
    )

    resp = requests.get(url)
    data_json = resp.json()

    # --- Validate response ---
    if "values" not in data_json or not data_json["values"]:
        raise ValueError(f"Twelve Data returned no usable data for {ticker}: {data_json}")

    # --- Convert to DataFrame ---
    df = pd.DataFrame(data_json["values"])
    df.rename(
        columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )

    # Convert to numeric types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # Some crypto pairs don't return volume — fill with zeros
            df[col] = 0

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Clip date range
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    if df.empty:
        raise ValueError(f"No data found for {ticker} between {start} and {end}.")

    df.attrs["market_type"] = "crypto" if is_crypto else "stock"
    return df.dropna()

# ------------------ TECHNICAL FEATURES ------------------
def _technical_features(df):
    df = df.copy()
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)
    df["Lag3"] = df["Return"].shift(3)
    df["MA5"] = df["Close"].rolling(5).mean() / df["Close"] - 1
    df["Vol10"] = df["Return"].rolling(10).std()
    return df


import requests

FMP_API_KEY = "soFZaj9OS0tmLv0UzAk2YACP1bIl4fie"

# ------------------ FUNDAMENTALS ------------------
def _fetch_fundamentals(ticker):
    try:
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=4&apikey={FMP_API_KEY}"
        r = requests.get(url)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        keep = ["date", "revenue", "netIncome", "eps", "grossProfitRatio"]
        df = df[[c for c in keep if c in df.columns]]
        df.rename(columns={"date": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        print(f"⚠️ FMP fundamentals fetch failed: {e}")
        return pd.DataFrame()


# ------------------ NEWS SENTIMENT ------------------
def _fetch_news_sentiment(ticker):
    try:
        import requests
        from transformers import pipeline

        FMP_API_KEY = "soFZaj9OS0tmLv0UzAk2YACP1bIl4fie"
        url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=50&apikey={FMP_API_KEY}"
        r = requests.get(url)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()

        sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

        sentiments = []
        for item in data:
            title = item.get("title", "")
            date = item.get("publishedDate", "")
            if title:
                score = sentiment_analyzer(title)[0]
                sentiments.append({
                    "Date": pd.to_datetime(date.split("T")[0]),
                    "Sentiment": score["label"],
                    "Confidence": score["score"]
                })

        df = pd.DataFrame(sentiments)
        if not df.empty:
            df["SentimentScore"] = df["Sentiment"].map({
                "positive": 1, "neutral": 0, "negative": -1
            }) * df["Confidence"]
            df = df.groupby("Date")["SentimentScore"].mean().reset_index()

        return df
    except Exception as e:
        print(f"⚠️ FinBERT sentiment fetch failed: {e}")
        return pd.DataFrame()



# ------------------ MACRO DATA ------------------
from fredapi import Fred

def _fetch_macro_data():
    try:
        fred = Fred(api_key=FRED_API_KEY)
        cpi = fred.get_series("CPIAUCSL").pct_change().rename("CPI_Change")
        fedfunds = fred.get_series("FEDFUNDS").rename("FedFunds")
        vix = fred.get_series("VIXCLS").rename("VIX")
        macro = pd.concat([cpi, fedfunds, vix], axis=1)
        macro = macro.ffill().dropna().reset_index()
        macro.rename(columns={"index": "Date"}, inplace=True)
        return macro
    except Exception as e:
        print(f"⚠️ FRED macro fetch failed: {e}")
        return pd.DataFrame()


# ------------------ MARKET CONTEXT DATA (STOCKS + CRYPTO) ------------------
def _fetch_market_context(market_type="stock", start=None, end=None):
    """
    Fetches external context features:
      - Stocks: macro (FRED)
      - Crypto: market sentiment (Fear & Greed) + BTC dominance + total market cap
    """
    if market_type == "stock":
        try:
            fred = Fred(api_key=FRED_API_KEY)
            cpi = fred.get_series("CPIAUCSL").pct_change().rename("CPI_Change")
            fedfunds = fred.get_series("FEDFUNDS").rename("FedFunds")
            vix = fred.get_series("VIXCLS").rename("VIX")
            macro = pd.concat([cpi, fedfunds, vix], axis=1)
            macro = macro.ffill().dropna().reset_index()
            macro.rename(columns={"index": "Date"}, inplace=True)
            macro["Date"] = pd.to_datetime(macro["Date"])
            macro = macro.set_index("Date").asfreq("D").ffill().reset_index()
            if start and end:
                macro = macro.loc[
                    (macro["Date"] >= pd.Timestamp(start)) &
                    (macro["Date"] <= pd.Timestamp(end))
                ]
            return macro

        except Exception as e:
            print(f"⚠️ FRED macro fetch failed: {e}")
            return pd.DataFrame()

    else:
        # ----- CRYPTO MARKET CONTEXT -----
        try:
            fng_url = "https://api.alternative.me/fng/?limit=0"
            fng_resp = requests.get(fng_url).json()
            fng = pd.DataFrame(fng_resp["data"])
            fng["Date"] = pd.to_datetime(fng["timestamp"], unit="s")
            fng["FearGreed"] = pd.to_numeric(fng["value"], errors="coerce")
            fng = fng[["Date", "FearGreed"]]

            cg_url = "https://api.coingecko.com/api/v3/global"
            cg_resp = requests.get(cg_url).json()["data"]
            mcap = pd.DataFrame({
                "Date": [pd.Timestamp.utcnow().normalize()],
                "BTC_Dominance": [cg_resp["market_cap_percentage"]["btc"]],
                "Total_Market_Cap": [cg_resp["total_market_cap"]["usd"]],
            })

            crypto_macro = pd.merge_asof(
                fng.sort_values("Date"),
                mcap.sort_values("Date"),
                on="Date",
                direction="backward",
            )

            if start and end:
                crypto_macro = crypto_macro.loc[
                    (crypto_macro["Date"] >= pd.Timestamp(start)) &
                    (crypto_macro["Date"] <= pd.Timestamp(end))
                ]

            return crypto_macro.ffill().dropna()

        except Exception as e:
            print(f"⚠️ Crypto macro fetch failed: {e}")
            return pd.DataFrame()


# ------------------ PREPARE FEATURES ------------------
def prepare_features(data, include_extra=False, ticker=None):
    """
    Prepare model-ready features for both stocks and crypto.
    Adds technical indicators, and optionally merges fundamentals/news/macro (stocks)
    or crypto context metrics (Fear & Greed, BTC dominance, etc).
    """
    df = _technical_features(data).dropna().copy()
    market_type = data.attrs.get("market_type", "stock")

    # --- Extended Data ---
    if include_extra and ticker:
        print(f"📡 Fetching extended data for {market_type.upper()}...")

        if market_type == "stock":
            # For stocks → fundamentals + news + macro
            fundamentals = _fetch_fundamentals(ticker)
            sentiment = _fetch_news_sentiment(ticker)
            macro = _fetch_market_context("stock", start=df.index.min(), end=df.index.max())

            for extra_df in [fundamentals, sentiment, macro]:
                if not extra_df.empty and "Date" in extra_df.columns:
                    df = df.merge(extra_df, on="Date", how="left")

        elif market_type == "crypto":
            # For crypto → global sentiment & market metrics
            crypto_context = _fetch_market_context("crypto", start=df.index.min(), end=df.index.max())
            if not crypto_context.empty and "Date" in crypto_context.columns:
                df = df.reset_index().merge(crypto_context, on="Date", how="left").set_index("Date")

        df = df.fillna(method="ffill").fillna(0)
        print("✅ Enhanced feature set ready.")

    return df.dropna()


# ------------------ SPLIT DATA ------------------
def split_data(data, train_ratio=0.8):
    """
    Split the data into training and test sets safely.
    Automatically rebuilds any missing technical indicators
    (Lag, MA5, Vol10) before training.
    """
    import numpy as np
    import pandas as pd

    df = data.copy()

    # --- Ensure required base column ---
    if "Close" not in df.columns:
        raise KeyError("Missing 'Close' column — cannot rebuild features.")

    # --- Rebuild Return if missing ---
    if "Return" not in df.columns:
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # --- Rebuild technicals if missing ---
    if "Lag1" not in df.columns:
        df["Lag1"] = df["Return"].shift(1)
    if "Lag2" not in df.columns:
        df["Lag2"] = df["Return"].shift(2)
    if "Lag3" not in df.columns:
        df["Lag3"] = df["Return"].shift(3)
    if "MA5" not in df.columns:
        df["MA5"] = df["Close"].rolling(5).mean() / df["Close"] - 1
    if "Vol10" not in df.columns:
        df["Vol10"] = df["Return"].rolling(10).std()

    # --- Drop rows with NaNs after rebuilding ---
    df.dropna(inplace=True)

    # --- Define features and target ---
    features = ["Lag1", "Lag2", "Lag3", "MA5", "Vol10"]
    X = df[features]
    y = df["Return"]

    # --- Split train/test ---
    split_idx = int(len(df) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_data = df.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, test_data

# ------------------ BACKTEST ------------------
def backtest_strategy(test_data, predictions):
    realized_returns = test_data["Return"].values
    signals = np.where(predictions > 0, 1, -1)
    n = min(len(signals), len(realized_returns))
    strat_returns = signals[:n] * realized_returns[:n]
    cumulative = np.cumsum(strat_returns)
    sharpe = np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(252) if np.std(strat_returns) else 0
    win_rate = np.mean(strat_returns > 0)
    return cumulative, sharpe, win_rate
# ------------------ TRAIN AND EVALUATE ------------------
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Fit the model and compute R² and MSE metrics.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    return preds, r2, mse
# ------------------ FORECAST FUTURE PRICES ------------------
def forecast_future_prices(model, data, days_ahead=5):
    """
    Forecast future stock or crypto prices using the trained model and last available features.
    Handles missing columns gracefully and ensures continuous forecast dates.
    """
    import pandas as pd
    import numpy as np

    features = ["Lag1", "Lag2", "Lag3", "MA5", "Vol10"]
    df = data.copy()

    # --- Flatten MultiIndex if necessary ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # --- Handle missing features dynamically ---
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 3:
        print("⚠️ Too few available features for forecasting.")
        return pd.DataFrame()

    df = df.dropna(subset=available_features).tail(50)
    if len(df) < 10:
        print("⚠️ Not enough valid data for forecasting. Try a longer date range.")
        return pd.DataFrame()

    last_close = df["Close"].iloc[-1]
    forecasts = []

    for _ in range(days_ahead):
        feature_row = df[available_features].iloc[-1:]
        predicted_return = model.predict(feature_row)[0]
        next_price = last_close * np.exp(predicted_return)

        # --- Handle datetime safely ---
        last_date = df.index[-1]
        if not isinstance(last_date, pd.Timestamp):
            last_date = pd.Timestamp.today().normalize()
        next_date = last_date + pd.Timedelta(days=1)

        forecasts.append({
            "Date": next_date,
            "PredictedPrice": next_price,
            "PredictedReturn": predicted_return
        })

        # --- Append new synthetic data point ---
        new_row = pd.DataFrame({
            "Close": [next_price],
            "Return": [predicted_return],
            "Lag1": [df["Return"].iloc[-1]],
            "Lag2": [df["Lag1"].iloc[-1]] if "Lag1" in df else [0],
            "Lag3": [df["Lag2"].iloc[-1]] if "Lag2" in df else [0],
            "MA5": [df["Close"].rolling(5).mean().iloc[-1] / df["Close"].iloc[-1] - 1],
            "Vol10": [df["Return"].rolling(10).std().iloc[-1]] if "Return" in df else [0],
        }, index=[next_date])

        df = pd.concat([df, new_row])
        last_close = next_price

    print(f"✅ Forecast generated: {len(forecasts)} days")
    if len(forecasts) > 0:
        print("Preview →", forecasts[-1])

    # --- Final return ---
    return pd.DataFrame(forecasts)
