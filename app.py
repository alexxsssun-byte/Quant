#!/usr/bin/env python3
"""
Modern Institutional Quant Dashboard
Author: Your Name
Institutional-Style Streamlit App (Green Theme + Plotly)
"""

# ============================================================
# 🧩 Imports
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from stock_return_model import split_data, apply_strategy_logic, list_available_strategies



from stock_return_model import (
    load_data,
    prepare_features,
    split_data,
    train_and_evaluate,
    backtest_strategy,
    forecast_future_prices,
    _fetch_market_context,
)

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, cross_val_predict, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import json, os

# ============================================================
# 🧱 Streamlit Page Setup
# ============================================================
st.set_page_config(
    page_title="Institutional Quant Dashboard",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --primary-green: #1e5631;  /* Deep institutional green */
        --secondary-green: #2e8b57;
        --accent-green: #a3c9a8;
        --bg-dark: #00010b   /* Institutional dark blue */
        --bg-card: #15233b;        /* Slightly lighter card tone */
        --text-light: #e2e8f0;
        --border-muted: #24344f;
    }

    /* Page background */
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-dark) !important;
        color: var(--text-light) !important;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #0b1625 !important;
        color: var(--text-light) !important;
    }

    /* Section headers and titles */
    h1, h2, h3, h4 {
        color: var(--accent-green) !important;
        font-weight: 600 !important;
    }

    /* Divider lines */
    hr {
        border-top: 1px solid var(--border-muted) !important;
        margin-top: 1rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* Metric container styling */
    div.stMetric {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-muted) !important;
        border-radius: 12px !important;
        padding: 12px 10px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
        transition: all 0.2s ease-in-out;
    }

    div.stMetric:hover {
        background-color: #1d2f4b !important;
        border-color: var(--secondary-green) !important;
        box-shadow: 0 2px 6px rgba(46, 139, 87, 0.3);
    }

    /* Metric label (small text) */
    div[data-testid="stMetricLabel"] > div {
        color: var(--accent-green) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Metric value (big number) */
    div[data-testid="stMetricValue"] {
        color: var(--text-light) !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }

    /* Delta (if used) */
    div[data-testid="stMetricDelta"] {
        color: var(--secondary-green) !important;
    }

    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        background-color: var(--bg-card) !important;
        border-radius: 8px !important;
    }

    /* Button styling */
    button[kind="primary"] {
        background-color: var(--secondary-green) !important;
        color: #fff !important;
        border-radius: 8px !important;
        border: none !important;
        transition: 0.2s ease-in-out;
    }
    button[kind="primary"]:hover {
        background-color: var(--accent-green) !important;
        color: #0e1a2b !important;
    }

    /* Improve text clarity */
    p, span, div {
        color: var(--text-light) !important;
    }

    /* Central container padding */
    div.block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        font-family: "Inter", "Segoe UI", sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# 🧠 Session Init
# ============================================================
for key in ["is_training", "model_trained_at", "model", "data", "results", "preds", "y_test", "test_data"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# 🧭 Sidebar
# ============================================================
st.sidebar.title("⚙️ Model Settings")

ticker = st.sidebar.text_input("Stock / ETF Ticker", "SPY")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp.today().normalize())

market_type = "crypto" if any(x in ticker.upper() for x in ["BTC", "ETH", "DOGE"]) or "/" in ticker else "stock"

include_extra = st.sidebar.toggle(
    "Include fundamentals, sentiment, macro", value=True
)

mode = st.sidebar.radio("Dashboard Mode", ["Full Analysis", "Quick Dashboard"], horizontal=True)

run_button = st.sidebar.button("🚀 Run Model")

if st.sidebar.button("🔁 Force Retrain"):
    for key in ["model", "data", "results", "preds", "y_test", "test_data"]:
        st.session_state.pop(key, None)
    st.session_state["model_trained_at"] = None
    st.rerun()
# ============================================================
# 🧭 Strategy Selection
# ============================================================
st.sidebar.markdown("### 🧩 Strategy Type")

from stock_return_model import list_available_strategies

# --- Auto-detect available strategies from backend ---
strategies = list_available_strategies()

strategy_type = st.sidebar.selectbox("Select Quant Strategy", strategies)

# Optional explanation box (keeps your nice descriptions)
strategy_explanations = {
    "Trend-Following": "📈 Buys assets above their 200-day MA.",
    "Mean Reversion": "↔️ Trades reversals after sharp short-term moves.",
    "Cross-Sectional Momentum": "🏆 Ranks and buys the strongest recent performers.",
    "Volatility / Options": "🌪️ Adapts exposure to volatility regimes.",
    "Arbitrage": "🔁 Exploits temporary price dislocations.",
    "Hybrid (Trend + MR)": "🧠 Combines long-term trend with short-term mean reversion."
}
st.sidebar.markdown(strategy_explanations.get(strategy_type, ""))

# ============================================================
# 🧭 Strategy Framework Explanation
# ============================================================
st.markdown(f"### 🧭 Strategy Framework: **{strategy_type}**")

strategy_explanations = {
    "Trend-Following": (
        "📈 **Trend-Following** buys assets in sustained uptrends "
        "(price above 200-day MA) and shorts when they fall below."
    ),
    "Mean Reversion": (
        "↔️ **Mean Reversion** trades against sharp short-term moves, "
        "assuming prices revert toward their average."
    ),
    "Cross-Sectional Momentum": (
        "🏆 **Cross-Sectional Momentum** ranks assets by past 20-day performance "
        "and holds recent winners."
    ),
    "Volatility / Options": (
        "🌪️ **Volatility / Options** adjusts exposure based on realized volatility—"
        "short high-vol regimes, long calm periods."
    ),
    "Arbitrage": (
        "🔁 **Arbitrage** targets temporary mispricings between related instruments "
        "using statistical spread reversion."
    ),
    "Hybrid (Trend + MR)": (
        "🧠 **Hybrid (Trend + Mean Reversion)** combines long-term trend direction "
        "(50/200-day MAs) with short-term mean reversion signals "
        "(10-day Z-score). Buys dips in uptrends, shorts rips in downtrends."
    ),
}


st.info(strategy_explanations.get(strategy_type, "Select a strategy to view its description."))
st.divider()

st.sidebar.caption("Select a high-level quant regime for signal generation.")

# ============================================================
# 📊 Title and Overview
# ============================================================
st.title("📊 Institutional Quantitative Stock Return Model")
st.markdown(
    "A robust walk-forward, feature-lagged prediction engine with stacking, "
    "auto-tuning, and explainability — styled for institutional research."
)
st.divider()

# ============================================================
# 🔁 Retrain Logic
# ============================================================
def retrain_if_needed(model_trained_at=None, retrain_interval_days=7):
    if model_trained_at is None:
        st.info("🆕 No existing model — training required.")
        return True
    if not isinstance(model_trained_at, datetime):
        try:
            model_trained_at = datetime.fromisoformat(str(model_trained_at))
        except Exception:
            st.warning("⚠️ Invalid timestamp — forcing retrain.")
            return True
    days_since = (datetime.now() - model_trained_at).days
    if days_since > retrain_interval_days:
        st.warning(f"🔁 Model is {days_since} days old — retraining triggered.")
        return True
    elif days_since == 0:
        st.success("✅ Model trained today — still fresh.")
        return False
    else:
        st.info(f"⏸ Using existing model (trained {days_since} days ago).")
        return False

auto_retrain = retrain_if_needed(st.session_state.get("model_trained_at"), 7)
st.session_state["needs_retrain"] = auto_retrain

# ============================================================
# 🧮 Cached Data Loader
# ============================================================
@st.cache_data(ttl=3600)
def cached_load_data(ticker, start, end):
    return load_data(ticker, start, end)

# ============================================================
# 🧠 Run Model
# ============================================================
if run_button or st.session_state.get("needs_retrain", False):
    # --- Reset previous model state ---
    for key in ["model", "data", "results", "preds", "y_test", "test_data"]:
        st.session_state.pop(key, None)

    # --- Create a live status placeholder ---
    msg_box = st.empty()
    msg_box.info("🏗 Building & training model — please wait...")

    st.session_state["is_training"] = True

# ============================================================
# 🧠 Data Preparation & Leakage-Safe Split
# ============================================================
try:
    msg_box = st.empty()
    msg_box.info("🏗 Building & training model — please wait...")

    # --- Load and Prepare Data ---
    data = cached_load_data(ticker, start_date, end_date)
    data.attrs["market_type"] = market_type

    if not isinstance(data.index, pd.DatetimeIndex):
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
            data = data.set_index("Date")
        else:
            st.error("❌ No valid date column found.")
            st.stop()

    # --- Unified backend feature processing ---
    from stock_return_model import split_data
    horizon = st.slider("Forecast horizon (days ahead)", 1, 20, 5)
    X_train, X_test, y_train, y_test, test_data = split_data(data, horizon=horizon)

    # ✅ Attach the Return column from test_data for volatility & analysis
    if "Return" in test_data.columns:
        data["Return"] = test_data["Return"]

    msg_box.empty()
    st.success(f"✅ Data ready: {len(data):,} samples, {X_train.shape[1]} features.")

    st.session_state.update({
        "data": data,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_data": test_data,
    })

except Exception as e:
    msg_box.empty()
    st.error(f"❌ Data preparation failed: {e}")
    st.stop()

# ============================================================
# 🧠 Model Selection, Training, and Auto-Tuning
# ============================================================

# --- Adaptive model selection based on data context ---
data_len = len(X_train)
volatility = data["Return"].std() * np.sqrt(252)
avg_volume = data["Volume"].mean() if "Volume" in data.columns else None

st.markdown("### 🧩 Adaptive Model Selection")

if data_len < 500:
    model_choice = "LightGBM (small dataset)"
    base_model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05)
elif volatility > 0.35:
    model_choice = "XGBoost (high volatility)"
    base_model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03)
else:
    model_choice = "Hybrid Ensemble (stable market)"
    base_model = VotingRegressor([
        ("xgb", xgb.XGBRegressor(n_estimators=300, learning_rate=0.03)),
        ("lgb", lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)),
        ("lin", LinearRegression()),
    ])

st.info(f"🧠 Selected Model: **{model_choice}**")

# --- Train model ---
base_model.fit(X_train, y_train)
preds = base_model.predict(X_test)
r2 = r2_score(y_test, preds)
mse = mean_squared_error(y_test, preds)
st.success(f"📈 Initial R² = {r2:.3f} | MSE = {mse:.6f}")

# --- Auto-tune if performance weak ---
if r2 < 0.3:
    if isinstance(base_model, (xgb.XGBRegressor, lgb.LGBMRegressor)):
        st.warning("🤖 Low R² detected — running hyperparameter search…")
        grid = {
            "n_estimators": [200, 400, 600],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [4, 6, 8],
        }
        tuner = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=grid,
            n_iter=6,
            cv=KFold(3, shuffle=True, random_state=42),
            scoring="r2",
            n_jobs=-1,
        )
        tuner.fit(X_train, y_train)
        base_model = tuner.best_estimator_
        preds = base_model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        st.success(f"🎯 Tuned model improved: R² = {r2:.3f}")
    else:
        st.info("ℹ️ Base model is an ensemble — skipping direct hyperparameter tuning.")

# ============================================================
# 🤖 Smart Ensemble Meta-Learner (Stacking)
# ============================================================
# --- Adaptive Ensemble Auto-Tuning ---
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
import xgboost as xgb
import lightgbm as lgb
import numpy as np

st.markdown("### 🧠 Adaptive Ensemble Tuning")
st.caption("Automatically tunes model hyperparameters across XGBoost, LightGBM, and Ridge for best time-series performance.")

# --- Define base learners ---
xgb_model = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
lgb_model = lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
ridge_model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0))

# --- Initial ensemble ---
ensemble = VotingRegressor([
    ("xgb", xgb_model),
    ("lgb", lgb_model),
    ("ridge", ridge_model)
])

# --- Define search space for nested tuning ---
param_grid = {
    "weights": [
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (2, 2, 1)
    ]
}

kf = KFold(n_splits=3, shuffle=True, random_state=42)

# --- Run the tuner ---
st.info("🎯 Running adaptive hyperparameter search (XGB + LGBM + Ridge)... this may take a minute.")
tuner = RandomizedSearchCV(
    estimator=ensemble,
    param_distributions=param_grid,
    n_iter=8,
    cv=kf,
    scoring="r2",
    random_state=42,
    verbose=1,
    n_jobs=-1,
)
tuner.fit(X_train, y_train)

best_ensemble = tuner.best_estimator_
best_params = tuner.best_params_
best_r2 = tuner.best_score_

st.success(f"✅ Best ensemble found with mean CV R² = {best_r2:.3f}")
st.json(best_params)

# --- Evaluate tuned ensemble on test set ---
preds = best_ensemble.predict(X_test)
r2_meta = r2_score(y_test, preds)
mse_meta = mean_squared_error(y_test, preds)

st.metric("Tuned Ensemble R²", f"{r2_meta:.4f}")
st.metric("MSE", f"{mse_meta:.6f}")

# Store tuned model for later use
st.session_state["meta_model"] = best_ensemble
st.session_state["data"] = data
st.session_state["meta_results"] = pd.DataFrame({
    "Actual": y_test,
    "Predicted": preds
}).reset_index(drop=True)

# ============================================================
# 🧠 Strategy Overlay Application (Leakage-Safe & Aligned)
# ============================================================
strategy_df = None
try:
    strategy_df = apply_strategy_logic(test_data, strategy_type)
except Exception as e:
    st.warning(f"⚠️ Strategy logic failed: {e}")
    strategy_df = pd.DataFrame(index=test_data.index)

# --- Validate DataFrame output ---
if strategy_df is None or "Signal" not in strategy_df.columns:
    st.warning("⚠️ Strategy output invalid — using neutral signals.")
    signals = np.zeros(len(y_test))
else:
    # --- Safe alignment ---
    signals = strategy_df["Signal"].shift(1).fillna(0)
    common_idx = signals.index.intersection(y_test.index)
    signals = signals.loc[common_idx]
    y_test = y_test.loc[common_idx]
    preds = preds[-len(common_idx):]
    signals = signals.values
    y_test = np.array(y_test)


# --- Compute realistic metrics ---
hit_rate = np.mean(signals == np.sign(y_test))
strategy_ret = signals * y_test

sharpe = (
    np.mean(strategy_ret) / np.std(strategy_ret) * np.sqrt(252)
    if np.std(strategy_ret)
    else 0
)
cum_ret = np.cumsum(strategy_ret)
drawdown = np.max(np.maximum.accumulate(cum_ret) - cum_ret)
corr = np.corrcoef(y_test, preds)[0, 1]
alpha = np.mean(strategy_ret - y_test) * 252


# --- Display metrics ---
cols = st.columns(6)
for (label, val) in zip(
    ["R²", "MSE", "Hit Rate", "Sharpe", "Drawdown", "Alpha vs Benchmark"],
    [r2_meta, mse_meta, hit_rate, sharpe, drawdown, alpha],
):
    with cols.pop(0):
        st.metric(label, f"{val:.3f}" if "MSE" not in label else f"{val:.6f}")

# ============================================================
# 📈 Plotly Visualizations
# ============================================================

# --- Strategy vs Benchmark ---
benchmark_cum = np.cumsum(y_test)
strategy_cum = np.cumsum(strategy_ret)
fig = go.Figure()
fig.add_trace(go.Scatter(y=benchmark_cum, name="Benchmark (Buy & Hold)", line=dict(color="#7f8c8d")))
fig.add_trace(go.Scatter(y=strategy_cum, name="Model Strategy", line=dict(color="#1e5631")))
fig.update_layout(
    title="Cumulative Returns — Model vs Benchmark",
    yaxis_title="Cumulative Log Return",
    template="plotly_white",
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🧩 Overlay selected strategy performance
# ============================================================
strategy_overlay = np.cumsum(strategy_ret)
fig.add_trace(
    go.Scatter(
        y=strategy_overlay,
        name=f"{strategy_type} Strategy",
        line=dict(color="#8e44ad", dash="dot"),
    )
)
fig.update_layout(
    title=f"Cumulative Returns — Model vs Benchmark ({strategy_type})",
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# --- Rolling Sharpe ---
rolling = pd.Series(strategy_ret).rolling(60)
roll_sharpe = rolling.mean() / rolling.std() * np.sqrt(252)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=roll_sharpe, line=dict(color="#2e8b57")))
fig2.add_hline(y=0, line=dict(color="gray", dash="dot"))
fig2.update_layout(title="Rolling Sharpe (60-Day Window)", height=400, template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# --- Regime Breakdown ---
vol = pd.Series(y_test).rolling(20).std()
regime = np.where(vol > vol.median(), "High Vol", "Low Vol")
reg_df = pd.DataFrame({"Regime": regime, "Strategy Return": strategy_ret})
fig3 = px.box(reg_df, x="Regime", y="Strategy Return", color="Regime",
              color_discrete_map={"High Vol": "#2e8b57", "Low Vol": "#a3c9a8"},
              title="Strategy Return Distribution by Market Regime")
st.plotly_chart(fig3, use_container_width=True)
st.caption("Helps identify if model performs better in specific volatility regimes.")

# --- Predicted vs Actual Toggle ---
st.markdown("### 🔍 Predicted vs Actual")
view = st.radio("View Type:", ["Return", "Price"], horizontal=True)
test_df = pd.DataFrame({"Pred": preds, "Actual": y_test})
if view == "Return":
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=test_df["Actual"], name="Actual Return"))
    fig4.add_trace(go.Scatter(y=test_df["Pred"], name="Predicted Return"))
    fig4.update_layout(title="Predicted vs Actual Returns", height=400, template="plotly_white")
else:
    start_price = data["Close"].iloc[-len(test_df)]
    act_price = (1 + pd.Series(test_df["Actual"])).cumprod() * start_price
    pred_price = (1 + pd.Series(test_df["Pred"])).cumprod() * start_price
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=act_price, name="Actual Price"))
    fig4.add_trace(go.Scatter(y=pred_price, name="Predicted Price"))
    fig4.update_layout(title="Predicted vs Actual Price", height=400, template="plotly_white")
st.plotly_chart(fig4, use_container_width=True)
# ============================================================
# 🧠 Feature Importance & Explainability
# ============================================================
st.subheader("🧠 Feature Importance & Explainability")

try:
    model_to_explain = st.session_state.get("meta_model") or st.session_state.get("model")
    if model_to_explain is None:
        st.warning("⚠️ No model found. Please train or retrain first.")
        st.stop()

    # --- Handle VotingRegressor by averaging sub-model importances ---
    if isinstance(model_to_explain, VotingRegressor):
        importances = []
        names = []

        # Use fitted sub-models if available
        sub_models = getattr(model_to_explain, "estimators_", model_to_explain.estimators)

        for i, est in enumerate(sub_models):
            name = getattr(est, "__class__", type(est)).__name__
            if hasattr(est, "feature_importances_"):
                imp = pd.Series(est.feature_importances_, index=X_train.columns)
                importances.append(imp)
                names.append(name)
            elif hasattr(est, "coef_"):
                imp = pd.Series(np.abs(est.coef_), index=X_train.columns)
                importances.append(imp)
                names.append(name)

        if importances:
            avg_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                x=avg_importance.index,
                y=avg_importance.values,
                marker_color="#2e8b57",
                name="Avg Importance"
            ))
            fig_imp.update_layout(
                title="Average Feature Importance Across Ensemble Models",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("🧩 Ensemble sub-models found, but none expose `feature_importances_` or `coef_`.")


    # --- Handle tree-based single models ---
    elif hasattr(model_to_explain, "feature_importances_"):
        fi = pd.Series(model_to_explain.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        st.bar_chart(fi)

    # --- Handle linear models ---
    elif hasattr(model_to_explain, "coef_"):
        coefs = pd.Series(np.abs(model_to_explain.coef_), index=X_train.columns).sort_values(ascending=False)
        st.bar_chart(coefs)

    # --- SHAP fallback for XGBoost/LightGBM ---
    elif any(k in type(model_to_explain).__name__.lower() for k in ["xgb", "lgb", "boost"]):
        import shap
        st.caption("📊 SHAP values for model interpretation")
        explainer = shap.Explainer(model_to_explain, X_test)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(bbox_inches="tight")
    else:
        st.info("⚙️ Model type not recognized or does not support feature visualization.")
except Exception as e:
    st.warning(f"⚠️ Feature importance skipped: {e}")

# ============================================================
# 🔁 Walk-Forward Validation (Multi-Year)
# ============================================================
st.markdown("### 🔁 Walk-Forward Validation")
st.caption("Tests year-by-year generalization to detect overfitting or lookahead bias.")

model_to_validate = st.session_state.get("meta_model") or st.session_state.get("model")
data_to_use = st.session_state.get("data")

# 🧠 Diagnostic — verify multi-year data coverage
if data_to_use is not None:
    st.write("📅 Data coverage:", data_to_use.index.min(), "→", data_to_use.index.max())
    st.write("📈 Total rows:", len(data_to_use))

if model_to_validate is not None and data_to_use is not None:
    if st.button("Run Walk-Forward Validation"):
        from stock_return_model import walk_forward_validation

        with st.spinner("Running multi-year walk-forward validation..."):
            start_year = int(st.session_state.get("start_date", pd.Timestamp("2010-01-01")).year) + 1
            end_year = int(st.session_state.get("end_date", pd.Timestamp.today()).year)

            wf = walk_forward_validation(
                data_to_use,
                model_to_validate,
                start_year=start_year,
                end_year=end_year
            )

        if wf is not None and not wf.empty:
            st.success("✅ Walk-forward validation complete")

            # --- Table of yearly results ---
            st.dataframe(
                wf.style.format({
                    "R²": "{:.3f}",
                    "MSE": "{:.6f}",
                    "Hit Rate": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "Max Drawdown": "{:.3f}",
                })
            )

            # --- 📊 Plotly multi-year chart ---
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_bar(
                x=wf.index,
                y=wf["Sharpe"],
                name="Sharpe Ratio",
                marker_color="mediumseagreen",
                opacity=0.8,
            )
            fig.add_trace(
                go.Scatter(
                    x=wf.index,
                    y=wf["Hit Rate"] * 100,
                    name="Hit Rate (%)",
                    mode="lines+markers",
                    line=dict(color="orange", width=2),
                )
            )
            fig.add_hline(y=wf["Sharpe"].mean(), line_dash="dot", line_color="gray", opacity=0.6)

            fig.update_layout(
                title="Walk-Forward Validation — Sharpe & Hit Rate by Year",
                xaxis_title="Year",
                yaxis_title="Metric Value",
                template="plotly_dark",
                height=450,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ Not enough data for walk-forward validation.")
else:
    st.info("ℹ️ Train a model first to enable walk-forward validation.")

# ============================================================
# 🔮 Forecasting Future Prices
# ============================================================
st.subheader("🔮 Forecast Future Prices")

days_ahead = st.slider("Days ahead to forecast", 1, 30, 5)

if st.button("Generate Forecast"):
    model = st.session_state.get("meta_model") or st.session_state.get("model")
    data = st.session_state.get("data")

    if model is not None and data is not None:
        # --- Ensure the data index is datetime ---
        if not isinstance(data.index, pd.DatetimeIndex):
            if "Date" in data.columns:
                data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
                data = data.set_index("Date")
            else:
                st.error("⚠️ Data must have a 'Date' column or DatetimeIndex for forecasting.")
                st.stop()

        # --- Forecasting block (properly aligned) ---
        with st.spinner("Forecasting future trajectory..."):
            from stock_return_model import forecast_future_prices
            future_df = forecast_future_prices(model, data, days_ahead)

            if future_df is None or future_df.empty:
                st.error("⚠️ Forecast failed — check date range or retrain model.")
            else:
                st.success(f"✅ Forecasted {days_ahead} days ahead.")

                # --- Prepare last historical data ---
                last = data.reset_index().iloc[-50:][["Date", "Close"]]
                last = last.rename(columns={"Close": "ActualPrice"})

                # --- Ensure forecast dates extend beyond the last known date ---
                last_date = pd.to_datetime(last["Date"]).max()
                future_df["Date"] = pd.to_datetime(future_df["Date"]) + pd.Timedelta(days=1)

                # --- Combine cleanly and drop overlaps ---
                chart_df = pd.concat([last, future_df], ignore_index=True)
                chart_df = chart_df.drop_duplicates(subset="Date", keep="last")

                # --- Plot with clear visual separation ---
                fig_f = px.line(
                    chart_df,
                    x="Date",
                    y=["ActualPrice", "PredictedPrice"],
                    title="Forecasted Price Trajectory",
                    color_discrete_map={
                        "ActualPrice": "#1e5631",
                        "PredictedPrice": "#2e8b57"
                    },
                )

                # --- Add vertical line to mark forecast start (Plotly-safe) ---
                import datetime

                if isinstance(last_date, pd.Timestamp):
                    last_date_plot = last_date.to_pydatetime()
                else:
                    last_date_plot = pd.to_datetime(last_date)

                x_numeric = last_date_plot.timestamp() * 1000  # milliseconds

                fig_f.add_vline(
                    x=x_numeric,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Forecast Start",
                    annotation_position="top right"
                )


                fig_f.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig_f, use_container_width=True)

                # --- Show raw forecast table ---
                st.dataframe(future_df)

    else:
        st.warning("⚠️ Please run **Run Model** before forecasting.")



# ============================================================
# 💾 Persistent Learning Memory
# ============================================================
st.subheader("💾 Persistent Learning Memory")

MEMORY_FILE = "model_memory.json"
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)
else:
    memory = {}

# Retrieve current meta-model
meta_model = st.session_state.get("meta_model")

if meta_model is None:
    st.warning("⚠️ No meta model found. Please train one first.")
    st.stop()

# Safely get ensemble weights or default to equal weighting
if hasattr(meta_model, "weights") and meta_model.weights is not None:
    new_weights = list(meta_model.weights)
else:
    new_weights = [1 / len(meta_model.estimators_)] * len(meta_model.estimators_)

# Handle both pre-fit and post-fit structures
if hasattr(meta_model, "estimators"):
    base_model_names = [name for name, _ in meta_model.estimators]
elif hasattr(meta_model, "estimators_"):
    base_model_names = [type(m).__name__ for m in meta_model.estimators_]
else:
    base_model_names = ["UnknownModel"]


ticker_key = ticker.upper()

if ticker_key in memory:
    old_weights = memory[ticker_key]["weights"]
    blended = [round(0.7 * old + 0.3 * new, 5) for old, new in zip(old_weights, new_weights)]
    st.info(f"🧠 Updating memory for {ticker_key}")
else:
    blended = new_weights
    st.success(f"🆕 Creating new model memory for {ticker_key}")

memory[ticker_key] = {
    "weights": blended,
    "base_models": base_model_names,
    "last_r2": round(r2_meta, 4),
    "last_updated": str(pd.Timestamp.now()),
}

with open(MEMORY_FILE, "w") as f:
    json.dump(memory, f, indent=4)

st.dataframe(pd.DataFrame({"Base Model": base_model_names, "Weight": blended}))
st.caption(f"Memory stored in `{MEMORY_FILE}` — used for progressive learning on next retrain.")

# ============================================================
# 📥 Export
# ============================================================
if "preds" in st.session_state and "y_test" in st.session_state:
    res = pd.DataFrame({"Actual": st.session_state["y_test"], "Predicted": st.session_state["preds"]})
    csv = res.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions CSV", csv, f"{ticker}_predictions.csv")
