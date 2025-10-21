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
from stock_return_model import load_data, prepare_features, split_data, train_and_evaluate, backtest_strategy, forecast_future_prices,   _fetch_market_context
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

@st.cache_data(ttl=3600)
def cached_load_data(ticker, start, end):
    """
    Cached wrapper for load_data() to avoid hitting Alpha Vantage limits.
    TTL = 3600 seconds (1 hour)
    """
    from stock_return_model import load_data
    return load_data(ticker, start, end)

st.set_page_config(page_title="Quant Stock Return Model", page_icon="📈", layout="centered")

# --- Sidebar ---
st.sidebar.title("⚙️ Model Settings")

# Basic inputs
ticker = st.sidebar.text_input("Stock Ticker", "SPY")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp.today().normalize())


# --- Detect Market Type ---
market_type = "crypto" if any(x in ticker.upper() for x in ["BTC", "ETH", "DOGE"]) or "/" in ticker else "stock"

# --- Dynamic Toggle Based on Market ---
if market_type == "crypto":
    include_extra = st.sidebar.toggle("Include crypto sentiment data", value=True)
else:
    include_extra = st.sidebar.toggle("Include fundamentals, news sentiment, and macro data", value=True)

# --- Run Button ---
run_button = st.sidebar.button("Run Model 🚀")


# --- Title ---
st.title("📊 Quantitative Stock Return Prediction")
st.markdown("Predict next-day stock returns using a simple **linear regression model** based on historical data.")

# --- User Options (always visible before run) ---
st.markdown("### ⚙️ Model Options")

# Detect if the ticker is a crypto pair
market_type = "crypto" if "/" in ticker or ticker.upper().endswith("-USD") else "stock"
st.write(f"🧠 Detected market type: {market_type.upper()}")
    

# --- Run Model ---
if run_button:
    # Always wipe any old model from memory
    for key in ["model", "data", "results"]:
        st.session_state.pop(key, None)

    with st.spinner("Building model... please wait"):
        # --- Load data (supports crypto & stocks) ---
        data = cached_load_data(ticker, start=start_date, end=end_date)
        data.attrs["market_type"] = market_type

        # --- Prepare features (with optional extra data) ---
        data = prepare_features(data, include_extra=include_extra, ticker=ticker)

        # --- Split data ---
        X_train, X_test, y_train, y_test, test_data = split_data(data)

        # --- Import models ---
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestRegressor, VotingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV

        # --- Analyze dataset context ---
        data_length = len(X_train)
        volatility = data["Return"].std() * np.sqrt(252)
        avg_volume = data["Volume"].mean() if "Volume" in data.columns else None

        st.write(f"📊 Data Summary — {data_length:,} samples | Annualized volatility: {volatility:.2%}")

        # --- Adaptive model selection ---
        if data_length < 500:
            model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
            model_name = "LightGBM (small dataset)"
        elif volatility > 0.35:
            model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03)
            model_name = "XGBoost (high volatility)"
        else:
            model = VotingRegressor([
                ("xgb", xgb.XGBRegressor(n_estimators=300, learning_rate=0.03)),
                ("lgb", lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)),
                ("lin", LinearRegression())
            ])
            model_name = "Ensemble Blend (stable market)"

        st.write(f"🧩 Selected Model: **{model_name}**")

        # --- Train ---
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.session_state["model"] = model
        st.session_state["data"] = data
        st.session_state["results"] = pd.DataFrame({"Actual": y_test, "Predicted": preds}).reset_index(drop=True)

        st.success("✅ Model training complete!")

        # --- Candidate models ---
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            ),
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            ),
        }

        # --- Smart model selection ---
        if data_length < 500:
            chosen_model = "LightGBM"
            st.info("📉 Small dataset — using LightGBM (optimized for limited data)")
        elif volatility > 0.4:
            chosen_model = "XGBoost"
            st.info("⚡ High-volatility stock — using XGBoost (robust under noise)")
        elif avg_volume and avg_volume > 5e7:
            chosen_model = "Hybrid Ensemble"
            st.info("🏦 Stable large-cap — blending XGB + LGBM + Linear Regression")
            best_model = VotingRegressor([
                ("xgb", models["XGBoost"]),
                ("lgb", models["LightGBM"]),
                ("lin", models["Linear Regression"]),
            ])
        else:
            chosen_model = "Random Forest"
            st.info("⚙️ Balanced data — using Random Forest baseline")

        if chosen_model != "Hybrid Ensemble":
            best_model = models[chosen_model]

        # --- Initial training ---
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)

        st.write(f"🧠 Selected Model: **{chosen_model}** — R² = {r2:.3f}")

        # --- Self-improvement: Auto-tune if R² < 0.3 ---
        if r2 < 0.3:
            st.warning("🤖 Low accuracy detected — auto-tuning hyperparameters...")
            param_grid = {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [4, 6, 8],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
            }
            tuner = RandomizedSearchCV(
                estimator=best_model,
                param_distributions=param_grid,
                n_iter=8,
                cv=KFold(3, shuffle=True, random_state=42),
                scoring="r2",
                n_jobs=-1,
                verbose=0,
                random_state=42,
            )
            tuner.fit(X_train, y_train)
            best_model = tuner.best_estimator_
            preds = best_model.predict(X_test)
            r2 = r2_score(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            st.success(f"🧩 Auto-tuned model improved! New R²: {r2:.3f}")

        # --- Feature Importance ---
        st.subheader("📈 Feature Importance")
        if hasattr(best_model, "feature_importances_"):
            importance = pd.Series(best_model.feature_importances_, index=X_train.columns)
            st.bar_chart(importance.sort_values(ascending=False))
        elif hasattr(best_model, "coef_"):
            importance = pd.Series(best_model.coef_, index=X_train.columns)
            st.bar_chart(importance.abs().sort_values(ascending=False))
        else:
            st.write("Model does not support feature importance extraction.")

        # --- Store results ---
        st.session_state["model"] = best_model
        st.session_state["data"] = data
        st.session_state["results"] = pd.DataFrame({"Actual": y_test, "Predicted": preds}).reset_index(drop=True)

        # --- Metrics ---
        st.metric(label="R² Score", value=f"{r2:.4f}")
        st.metric(label="Mean Squared Error", value=f"{mse:.6f}")

        # --- Smart Ensemble Stacking ---
        st.subheader("🤖 Smart Ensemble Meta-Learner")

        # Define base learners
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.linear_model import LinearRegression, RidgeCV
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import KFold, cross_val_predict
        
        if include_extra:
            st.success("🔍 Enhanced AI Mode — including fundamentals, sentiment, and macro indicators.")
        else:
            st.info("⚡ Fast Mode — technical indicators only.")

        base_models = {
            "XGBoost": xgb.XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            ),
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
            ),
        }

        meta_features = pd.DataFrame(index=y_train.index)

        st.write("🔧 Training base models with cross-validation stacking...")

        # Generate out-of-fold predictions (meta features)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in base_models.items():
            preds = cross_val_predict(model, X_train, y_train, cv=kf, n_jobs=-1)
            meta_features[name] = preds
            st.write(f"✅ {name} trained for meta-ensemble.")

        # --- Meta learner (stacking layer) ---
        meta_model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0))
        meta_model.fit(meta_features, y_train)

        # Train base models fully
        for name, model in base_models.items():
            model.fit(X_train, y_train)

        # Create meta features for test data
        test_meta = pd.DataFrame({name: model.predict(X_test) for name, model in base_models.items()})
        final_preds = meta_model.predict(test_meta)

        # Evaluate performance
        r2_meta = r2_score(y_test, final_preds)
        mse_meta = mean_squared_error(y_test, final_preds)

        st.success(f"🧠 Meta-ensemble trained successfully! R² = {r2_meta:.3f} | MSE = {mse_meta:.6f}")
        
        # --- Model Evaluation ---
        st.subheader("📈 Model Performance")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**Mean Squared Error:** {mse:.6f}")

        # Align test data
        test_df = test_data.copy()
        test_df["Predicted_Return"] = preds
        test_df["Actual_Return"] = y_test.values

        # --- Toggle between return and price view ---
        st.subheader("🔍 Predicted vs Actual Visualization")
        view_type = st.radio(
            "Select view:",
            ["Return Comparison", "Price Comparison"],
            horizontal=True
        )

        import matplotlib.pyplot as plt

        if view_type == "Return Comparison":
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(test_df.index, test_df["Actual_Return"], label="Actual", linewidth=1.8)
            ax.plot(test_df.index, test_df["Predicted_Return"], label="Predicted", linewidth=1.8)
            ax.set_title("Predicted vs Actual Returns", fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Log Return")
            ax.legend()
            st.pyplot(fig)

        else:
            # Convert returns to cumulative prices
            price_df = test_df.copy()
            start_price = data["Close"].iloc[-len(test_df)]
            price_df["Actual_Price"] = (1 + price_df["Actual_Return"]).cumprod() * start_price
            price_df["Predicted_Price"] = (1 + price_df["Predicted_Return"]).cumprod() * start_price

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(price_df.index, price_df["Actual_Price"], label="Actual Price", linewidth=1.8)
            ax2.plot(price_df.index, price_df["Predicted_Price"], label="Predicted Price", linewidth=1.8)
            ax2.set_title("Predicted vs Actual Price", fontsize=14)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price")
            ax2.legend()
            st.pyplot(fig2)

        # --- Display Sample Predictions ---
        if "results" in st.session_state and not st.session_state["results"].empty:
            st.subheader("🔍 Sample Predicted vs Actual Returns")
            st.dataframe(st.session_state["results"].head(15))
        else:
            st.warning("No prediction results available to display yet. Run the model first.")

        # Compare base vs meta performance
        comparison = pd.DataFrame({
            "Model": list(base_models.keys()) + ["Meta-Ensemble"],
            "R²": [r2_score(y_test, model.predict(X_test)) for model in base_models.values()] + [r2_meta],
        })
        st.dataframe(comparison)

        # Store results
        st.session_state["meta_model"] = meta_model
        st.session_state["meta_results"] = pd.DataFrame({
            "Actual": y_test,
            "Predicted": final_preds
        }).reset_index(drop=True)

        import json
        import os

        # --- Memory persistence path ---
        MEMORY_FILE = "model_memory.json"

        # --- Load existing memory ---
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                model_memory = json.load(f)
        else:
            model_memory = {}

        # --- Update memory for this ticker ---
        ticker_key = ticker.upper()

        # Get new learned weights
        new_weights = meta_model.coef_.tolist()
        base_model_names = list(base_models.keys())

        # If ticker exists in memory, blend old and new weights (progressive learning)
        if ticker_key in model_memory:
            old_weights = model_memory[ticker_key]["weights"]
            blended_weights = [
                round(0.7 * old + 0.3 * new, 5)
                for old, new in zip(old_weights, new_weights)
            ]
            st.info(f"🧩 Updating learned weights for {ticker_key} (progressive averaging)")
        else:
            blended_weights = new_weights
            st.success(f"💾 Saving new model memory for {ticker_key}")

        # Save blended weights and metadata
        model_memory[ticker_key] = {
            "weights": blended_weights,
            "base_models": base_model_names,
            "last_r2": round(r2_meta, 4),
            "last_updated": str(pd.Timestamp.now()),
        }

        # Write memory back to disk
        with open(MEMORY_FILE, "w") as f:
            json.dump(model_memory, f, indent=4)

        # --- Display learned weights ---
        st.subheader("🧠 Persistent Learning Memory")
        memory_df = pd.DataFrame({
            "Base Model": base_model_names,
            "Weight": blended_weights
        })
        st.dataframe(memory_df)

        st.caption(f"Memory saved in `{MEMORY_FILE}` — next run for {ticker_key} will start from these learned weights.")




# --- Forecast Section ---
st.subheader("🔮 Forecast Future Prices")

days_ahead = st.slider("Days to forecast ahead", 1, 30, 5)

if st.button("Run Forecast"):
    if "model" in st.session_state and "data" in st.session_state:
        model = st.session_state["model"]
        data = st.session_state["data"]

        with st.spinner("Forecasting future prices..."):
            future_df = forecast_future_prices(model, data, days_ahead)
            st.success(f"✅ Forecasted {days_ahead} days ahead.")
            last_known = data.reset_index().iloc[-50:][["Date", "Close"]]
            merged = pd.concat([
                last_known.rename(columns={"Close": "PredictedPrice"}),
                future_df
            ])
            st.line_chart(merged.set_index("Date"))
            st.dataframe(future_df)
    else:
        st.warning("⚠️ Please run the model first before forecasting.")

# --- Download results safely ---
if "results" in st.session_state:
    results = st.session_state["results"]
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Predictions CSV",
        data=csv,
        file_name=f"{ticker}_predictions.csv"
    )
else:
    st.info("ℹ️ Run the model first to enable CSV download.")
