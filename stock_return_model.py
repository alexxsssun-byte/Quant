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

    # --- Check if data too short, then fall back to yfinance ---
    if len(df) < 1000 or df.index.min() > pd.Timestamp("2015-01-01"):
        print(f"⚠️ {ticker}: Twelve Data history too short ({len(df)} rows). Falling back to yfinance...")
        import yfinance as yf
        df_yf = yf.download(ticker.replace("/", "-"), start=start, end=end)
        if not df_yf.empty:
            df = df_yf.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
            df.index.name = "Date"
            print(f"✅ {ticker}: Loaded {len(df)} rows from yfinance ({df.index.min().date()} → {df.index.max().date()})")
        else:
            print("⚠️ yfinance fallback failed — keeping Twelve Data result.")

    if df.empty:
        raise ValueError(f"No data found for {ticker} between {start} and {end}.")

    df.attrs["market_type"] = "crypto" if is_crypto else "stock"
    return df.dropna()


# ============================================================
# 🧩 Enhanced Technical Feature Builder (leakage-safe)
# ============================================================
def _technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build robust, leakage-safe technical indicators.
    All rolling and lag features are shifted by 1 to prevent look-ahead bias.
    """
    import ta

    df = df.copy()
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # --- Lagged returns (1–5 days) ---
    for i in range(1, 6):
        df[f"Lag{i}"] = df["Return"].shift(i)

    # --- Moving averages and volatility (shifted to avoid leakage) ---
    df["MA5"] = (df["Close"].rolling(5).mean().shift(1) / df["Close"]) - 1
    df["MA10"] = (df["Close"].rolling(10).mean().shift(1) / df["Close"]) - 1
    df["Vol10"] = df["Return"].rolling(10).std().shift(1)
    df["Vol20"] = df["Return"].rolling(20).std().shift(1)
    df["AdjReturn"] = df["Return"] / (df["Vol10"] + 1e-6)

    # --- Momentum & oscillator features ---
    try:
        df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi().shift(1)
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd().shift(1)
        boll = ta.volatility.BollingerBands(df["Close"])
        df["BollWidth"] = (boll.bollinger_hband() - boll.bollinger_lband()).shift(1)
    except Exception as e:
        print(f"⚠️ TA feature computation failed: {e}")
        df["RSI14"], df["MACD"], df["BollWidth"] = np.nan, np.nan, np.nan

    # --- Volume change ---
    df["VolumeChange"] = df["Volume"].pct_change().shift(1)

    # --- Cleanup ---
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

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


# ============================================================
# 🧠 Time-based Split + Feature Scaling
# ============================================================
def split_data(data, train_ratio=0.8, horizon=5):
    """
    Time-based split with leakage-safe feature engineering.
    Predicts log return over a multi-day horizon (default = 5 days).
    Includes StandardScaler normalization for model stability.
    """
    from sklearn.preprocessing import StandardScaler

    df = data.copy()

    # --- Ensure datetime index ---
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        else:
            raise ValueError("Data must include a 'Date' column or DatetimeIndex.")

    # --- Build features safely ---
    df = _technical_features(df)

    # --- Define multi-day forward target ---
    df["Target"] = np.log(df["Close"].shift(-horizon) / df["Close"])
    df = df.dropna()

    # --- Feature selection ---
    feature_cols = [
        "Lag1", "Lag2", "Lag3", "MA5", "MA10",
        "Vol10", "Vol20", "AdjReturn", "RSI14",
        "MACD", "BollWidth", "VolumeChange",
    ]
    X = df[feature_cols]
    y = df["Target"]

    # --- Time-based train/test split ---
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_data = df.iloc[split_idx:].copy()

    # --- Feature scaling ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
)

    df.attrs["scaler"] = scaler

    # ✅ Add the Return column for downstream volatility & strategy analysis
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    test_data["Return"] = np.log(test_data["Close"] / test_data["Close"].shift(1))

    print(f"🧩 Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Horizon: {horizon}d")
    print(f"Train range: {X_train.index.min().date()} → {X_train.index.max().date()}")
    print(f"Test  range: {X_test.index.min().date()} → {X_test.index.max().date()}")

    # --- Return scaled features + dataframes ---
    return X_train_scaled, X_test_scaled, y_train, y_test, test_data

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
    Forecast future stock or crypto prices using the trained model and latest available features.
    Dynamically aligns feature sets to the model, rebuilds technicals each step, and smooths forecasts for realism.
    """
    import pandas as pd
    import numpy as np
    from stock_return_model import _technical_features  # ensure available

    # --- Copy and sanity check ---
    df = data.copy()
    if df.empty:
        print("❌ No data provided for forecasting.")
        return pd.DataFrame()

    # --- Flatten MultiIndex if necessary ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # --- Base technical feature set ---
    default_features = [
        "Lag1", "Lag2", "Lag3", "MA5", "MA10",
        "Vol10", "Vol20", "AdjReturn", "RSI14",
        "MACD", "BollWidth", "VolumeChange"
    ]

    # --- Try to infer actual model feature names ---
    try:
        if hasattr(model, "feature_names_in_"):
            model_features = list(model.feature_names_in_)
        elif hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_names_in_"):
            model_features = list(model.estimators_[0].feature_names_in_)
        else:
            model_features = default_features
    except Exception:
        model_features = default_features

    print(f"🧠 Using {len(model_features)} features for forecasting: {model_features}")

    # --- Ensure we have technicals built ---
    df = _technical_features(df)
    available_features = [f for f in model_features if f in df.columns]
    if len(available_features) < 3:
        print("⚠️ Missing technicals — rebuilding again.")
        df = _technical_features(df)
        available_features = [f for f in model_features if f in df.columns]

    # --- Drop incomplete rows and keep recent history ---
    df = df.dropna(subset=available_features).tail(100)
    if len(df) < 10:
        print("⚠️ Not enough valid rows for forecast.")
        return pd.DataFrame()

    last_close = df["Close"].iloc[-1]
    forecasts = []

    for i in range(days_ahead):
        # --- Recompute rolling features each iteration ---
        df = _technical_features(df)
        available_features = [f for f in model_features if f in df.columns]

        if not available_features or df.empty:
            print("⚠️ No available features for forecasting — breaking early.")
            break

        feature_row = df[available_features].iloc[-1:].copy()

        # --- Ensure all required columns are present ---
        for col in model_features:
            if col not in feature_row.columns:
                feature_row[col] = 0
        feature_row = feature_row[model_features].fillna(0)

        # --- Predict ---
        try:
            predicted_return = float(model.predict(feature_row)[0])
        except Exception as e:
            print(f"⚠️ Prediction failed at step {i}: {e}")
            continue

        # --- ⚙️ Realism tweak: smooth but preserve volatility scale ---
        recent_mean = df["Return"].iloc[-10:].mean() if "Return" in df else 0
        recent_std = df["Return"].iloc[-10:].std() if "Return" in df else 0.0005

        # Blend prediction and add noise scaled by recent vol
        predicted_return = (0.6 * predicted_return + 0.4 * recent_mean)
        predicted_return *= (1 + np.random.normal(0, 0.3 * recent_std))

        # --- Compute next date and price ---
        last_close = df["Close"].iloc[-1]
        last_date = df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        next_price = last_close * np.exp(predicted_return)

        # --- Save forecast ---
        forecasts.append({
            "Date": next_date,
            "PredictedPrice": next_price,
            "PredictedReturn": predicted_return,
        })

        # --- Append synthetic row for next iteration ---
        new_row = pd.DataFrame(
            {
                "Close": [next_price],
                "Return": [predicted_return],
                "Volume": [df["Volume"].iloc[-1] if "Volume" in df else 1e6],
            },
            index=[next_date],
        )
        df = pd.concat([df, new_row])


        # --- ⚙️ Realism tweak: smooth but preserve volatility scale ---
        recent_mean = df["Return"].iloc[-10:].mean() if "Return" in df else 0
        recent_std = df["Return"].iloc[-10:].std() if "Return" in df else 0.0005

        # Blend the model’s prediction with realized mean, then rescale by recent vol
        predicted_return = (0.6 * predicted_return + 0.4 * recent_mean)
        predicted_return *= (1 + np.random.normal(0, 0.5 * recent_std))


        # --- Compute next date and price ---
        last_close = df["Close"].iloc[-1]
        last_date = df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        next_price = last_close * np.exp(predicted_return)

        forecasts.append({
            "Date": next_date,
            "PredictedPrice": next_price,
            "PredictedReturn": predicted_return
        })

        # --- Append new synthetic data point ---
        new_row = pd.DataFrame({
            "Close": [next_price],
            "Volume": [df["Volume"].iloc[-1] if "Volume" in df else 1e6]
        }, index=[next_date])
        df = pd.concat([df, new_row])

    # ============================================================
    # ✅ Wrap-up
    # ============================================================
    if not forecasts:
        print("⚠️ No forecasts generated.")
        return pd.DataFrame()

    print(f"✅ Forecast generated for {len(forecasts)} days.")
    print("🔍 Preview:", forecasts[-1])

    return pd.DataFrame(forecasts)

# ------------------ WALK-FORWARD VALIDATION (Leakage-Safe) ------------------
def walk_forward_validation(
    data, model, feature_cols=None, start_year=None, end_year=None, shuffle_test=False
):
    """
    Leakage-safe walk-forward validation:
    - Predicts next-day returns (Target = Return.shift(-1))
    - Recreates lagged features year-by-year to avoid peeking
    - Performs yearly train/test splits across entire dataset
    """

    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score, mean_squared_error

    df = data.copy().sort_index()  # ⚠️ Removed global .dropna()

    # --- Ensure DatetimeIndex ---
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex for walk-forward validation.")

    # --- Define Target (next-day return) ---
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Target"] = df["Return"].shift(-1)

    # --- Define Features ---
    if feature_cols is None:
        df["Lag1"] = df["Return"].shift(1)
        df["Lag2"] = df["Return"].shift(2)
        df["Lag3"] = df["Return"].shift(3)
        df["MA5"] = df["Close"].rolling(5).mean().shift(1)
        df["Vol10"] = df["Return"].rolling(10).std().shift(1)
        feature_cols = ["Lag1", "Lag2", "Lag3", "MA5", "Vol10"]

    # --- Auto-detect full available range if not provided ---
    if start_year is None:
        start_year = df.index.year.min() + 1
    if end_year is None:
        end_year = df.index.year.max()

    print(f"🧭 Walk-forward validation: {start_year} → {end_year}")

    results = []

    # --- Walk-forward loop ---
    for year in range(start_year, end_year + 1):
        train = df[df.index.year < year].copy()
        test = df[df.index.year == year].copy()

        # ✅ Drop NaNs separately per split (don’t lose early years)
        train = train.dropna(subset=feature_cols + ["Target"])
        test = test.dropna(subset=feature_cols + ["Target"])

        if len(train) < 500 or len(test) < 150:
            print(f"⏭️ Skipping {year} (Train={len(train)}, Test={len(test)})")
            continue

        X_train, y_train = train[feature_cols], train["Target"]
        X_test, y_test = test[feature_cols], test["Target"]

        if shuffle_test:
            y_train = y_train.sample(frac=1, random_state=42).reset_index(drop=True)
            y_test = y_test.sample(frac=1, random_state=42).reset_index(drop=True)

        # --- Train model ---
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # --- Metrics ---
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        hit_rate = np.mean(np.sign(preds) == np.sign(y_test))

        strat_ret = np.sign(preds) * y_test
        sharpe = (
            np.mean(strat_ret) / np.std(strat_ret) * np.sqrt(252)
            if np.std(strat_ret)
            else 0
        )
        cum_ret = np.cumsum(strat_ret)
        drawdown = np.max(np.maximum.accumulate(cum_ret) - cum_ret)

        print(f"📅 {year}: R²={r2:.3f}, Hit={hit_rate:.2%}, Sharpe={sharpe:.2f}, n={len(test)}")

        results.append({
            "Year": year,
            "Samples": len(test),
            "R²": r2,
            "MSE": mse,
            "Hit Rate": hit_rate,
            "Sharpe": sharpe,
            "Max Drawdown": drawdown
        })

    results_df = pd.DataFrame(results).set_index("Year")

    print(f"✅ Walk-forward results across {len(results_df)} years ready.")
    return results_df


    # --- Walk-forward loop ---
    for year in range(start_year, end_year + 1):
        train = df[df.index.year < year]
        test = df[df.index.year == year]

        # Skip only if training data is unrealistically small
        if len(train) < 500:
            continue

        # Allow test sets as small as ~150 samples (≈ half a trading year)
        if len(test) < 150:
            continue

        X_train = train[feature_cols]
        y_train = train["Target"]
        X_test = test[feature_cols]
        y_test = test["Target"]

        if shuffle_test:
            y_train = y_train.sample(frac=1, random_state=42).reset_index(drop=True)
            y_test = y_test.sample(frac=1, random_state=42).reset_index(drop=True)

        # --- Train model ---
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # --- Metrics ---
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        pred_sign = np.sign(preds)
        true_sign = np.sign(y_test)
        hit_rate = np.mean(pred_sign == true_sign)

        strategy_ret = pred_sign * y_test
        sharpe = np.mean(strategy_ret) / np.std(strategy_ret) * np.sqrt(252) if np.std(strategy_ret) else 0
        cum_ret = np.cumsum(strategy_ret)
        drawdown = np.max(np.maximum.accumulate(cum_ret) - cum_ret)

        # --- Log per-year diagnostics ---
        print(f"📅 {year}: R²={r2:.3f}, Hit={hit_rate:.2%}, Sharpe={sharpe:.2f}, n={len(test)}")

        # --- Append result ---
        results.append({
            "Year": year,
            "Samples": len(test),
            "R²": r2,
            "MSE": mse,
            "Hit Rate": hit_rate,
            "Sharpe": sharpe,
            "Max Drawdown": drawdown
        })

    # --- Compile full results ---
    results_df = pd.DataFrame(results).set_index("Year")

    if results_df.empty:
        print("⚠️ No valid yearly results — check data range.")
    else:
        print(f"✅ Walk-forward results across {len(results_df)} years ready.")

    return results_df

# ============================================================
# 🧠 Safe Tuner Wrapper — auto-manages parallelism and memory
# ============================================================
import os, psutil, gc
from sklearn.model_selection import RandomizedSearchCV

def safe_tuner(estimator, param_distributions, n_iter=5, cv=3, scoring="r2", random_state=42):
    """
    Creates a RandomizedSearchCV with adaptive parallelism and memory safety.
    Prevents joblib crashes (TerminatedWorkerError) on low-memory systems.
    """

    # --- Detect system resources ---
    try:
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count(logical=True)
    except Exception:
        total_ram_gb = 8
        cpu_cores = os.cpu_count() or 4

    # --- Adaptive n_jobs ---
    if total_ram_gb < 8:
        n_jobs = 1
    elif total_ram_gb < 16:
        n_jobs = max(1, cpu_cores // 2)
    else:
        n_jobs = min(4, cpu_cores // 2)

    # --- Adaptive iteration scaling ---
    n_iter = min(n_iter, 5 if total_ram_gb < 16 else n_iter)

    # --- Log resource decision ---
    print(f"🧠 SafeTuner: Detected {cpu_cores} cores, {total_ram_gb:.1f} GB RAM → using n_jobs={n_jobs}")

    # --- Construct tuner ---
    tuner = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=2,
        error_score="raise",
    )

    return tuner

# ============================================================
# ⚙️ QUANT STRATEGY ENGINE
# ============================================================
def apply_strategy_logic(data: pd.DataFrame, strategy_type: str = "Trend-Following") -> pd.DataFrame:
    """
    Apply simplified quant logic overlays to existing price data.
    Returns a DataFrame with a 'Signal' column: +1 = long, -1 = short, 0 = neutral.
    """

    df = data.copy()
    if "Close" not in df.columns:
        raise ValueError("Data must include a 'Close' column for strategy logic.")

    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Signal"] = 0

    # --- 1️⃣ Trend-Following ---
    if strategy_type == "Trend-Following":
        df["MA200"] = df["Close"].rolling(200).mean()
        df["Signal"] = np.where(df["Close"] > df["MA200"], 1, -1)

    # --- 2️⃣ Mean Reversion ---
    elif strategy_type == "Mean Reversion":
        df["CumRet1d"] = df["Return"].rolling(1).sum()
        df["Signal"] = np.where(df["CumRet1d"] > 0.025, -1, 1)

    # --- 3️⃣ Cross-Sectional Momentum (single-asset placeholder) ---
    elif strategy_type == "Cross-Sectional Momentum":
        df["Mom20"] = df["Close"].pct_change(20)
        median_mom = df["Mom20"].median()
        df["Signal"] = np.where(df["Mom20"] > median_mom, 1, -1)

    # --- 4️⃣ Volatility / Options (simple realized vol regime) ---
    elif strategy_type == "Volatility / Options":
        df["Vol20"] = df["Return"].rolling(20).std()
        mean_vol = df["Vol20"].mean()
        df["Signal"] = np.where(df["Vol20"] > mean_vol, -1, 1)

    # --- 5️⃣ Arbitrage (statistical spread logic) ---
    elif strategy_type == "Arbitrage":
        df["Spread"] = df["Close"] - df["Close"].rolling(5).mean()
        df["Zscore"] = (df["Spread"] - df["Spread"].mean()) / df["Spread"].std()
        df["Signal"] = np.where(df["Zscore"] > 1, -1, np.where(df["Zscore"] < -1, 1, 0))

    # --- 6️⃣ Hybrid (Trend + Mean Reversion)---
    elif strategy_type == "Hybrid (Trend + MR)":
        # Long-term trend filter
        df["MA50"] = df["Close"].rolling(50).mean()
        df["MA200"] = df["Close"].rolling(200).mean()
        df["TrendSignal"] = np.where(df["MA50"] > df["MA200"], 1, -1)

        # Short-term mean reversion (Z-score of 10-day price)
        df["Zscore"] = (df["Close"] - df["Close"].rolling(10).mean()) / df["Close"].rolling(10).std()

        # Combine both: follow trend, fade extremes
        df["Signal"] = np.where(
            df["TrendSignal"] == 1,                     # Bullish regime
            np.where(df["Zscore"] < -1, 1, np.where(df["Zscore"] > 1, -1, 1)),  # Buy dips, short overbought
            np.where(df["Zscore"] > 1, -1, np.where(df["Zscore"] < -1, 1, -1))  # Bearish regime
        )

        # Optional: volatility filter (stay neutral in extreme vol)
        df["Vol20"] = df["Return"].rolling(20).std()
        vol_threshold = df["Vol20"].mean() * 2
        df.loc[df["Vol20"] > vol_threshold, "Signal"] = 0

    # --- ✅ Clean and return ---
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# ============================================================
# 🧭 Dynamic Strategy List Extractor
# ============================================================
import inspect, re

def list_available_strategies() -> list:
    """
    Automatically extracts all strategy names from apply_strategy_logic().
    Reads the source code to find 'if/elif strategy_type == "<name>"' entries.
    """
    try:
        src = inspect.getsource(apply_strategy_logic)
        strategies = re.findall(r'strategy_type == "([^"]+)"', src)
        # Deduplicate while preserving order
        seen = set()
        strategies = [s for s in strategies if not (s in seen or seen.add(s))]
        return strategies
    except Exception as e:
        print(f"⚠️ Could not parse strategies: {e}")
        # Fallback list if introspection fails
        return [
            "Trend-Following",
            "Mean Reversion",
            "Cross-Sectional Momentum",
            "Volatility / Options",
            "Arbitrage",
            "Hybrid (Trend + MR)",
        ]

