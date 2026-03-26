from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import InformerForPrediction

# ------------------------------
# Page Config (Public Version)
# ------------------------------
st.set_page_config(
    page_title="ETF Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Load Model & Data (Public Path)
# ------------------------------
MODEL_ROOT = Path("./model")

@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    model = InformerForPrediction.from_pretrained(str(MODEL_ROOT / "informer"))
    scaler = joblib.load(MODEL_ROOT / "scaler.joblib")
    meta = json.load(open(MODEL_ROOT / "training_meta.json"))
    df = pd.read_csv(MODEL_ROOT / "dataset_cache.csv", parse_dates=["Date"])
    return model, scaler, meta, df

try:
    model, scaler, meta, df = load_resources()
except Exception as e:
    st.error(f"Model failed to load: {str(e)}")
    st.stop()

# ------------------------------
# Config
# ------------------------------
FEATURE_COLS = meta["feature_cols"]
WINDOW_SIZE = meta["window_size"]
PRED_LEN = meta["pred_len"]

# ------------------------------
# Data Preprocessing
# ------------------------------
df = df.sort_values("Date").reset_index(drop=True)
df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
df[FEATURE_COLS] = df[FEATURE_COLS].ffill().fillna(0)

# ------------------------------
# 🔥 STABLE PREDICTION (PUBLIC SAFE)
# ------------------------------
def make_prediction():
    X = scaler.transform(df[FEATURE_COLS].values)
    ctx = X[-WINDOW_SIZE:]

    past_values = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0)
    past_time = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
    future_time = torch.zeros((1, PRED_LEN, len(FEATURE_COLS)), dtype=torch.float32)
    future_values = torch.zeros((1, PRED_LEN, 1), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        out = model(
            past_values=past_values,
            past_time_features=past_time,
            future_values=future_values,
            future_time_features=future_time,
            past_observed_mask=torch.ones_like(past_values)
        )

    return out.loc.squeeze().cpu().numpy()

# ------------------------------
# Run Prediction
# ------------------------------
pred_returns = make_prediction()
last_price = df["Close"].iloc[-1]
last_date = df["Date"].iloc[-1]

# Calculate future prices
future_dates = pd.bdate_range(last_date, periods=PRED_LEN+1, freq="B")[1:]
pred_prices = [last_price]
for r in pred_returns:
    pred_prices.append(pred_prices[-1] * (1 + float(r)))
pred_prices = np.array(pred_prices[1:])

# ------------------------------
# UI / PUBLIC DASHBOARD
# ------------------------------
st.title("📈 ETF Price Prediction Dashboard")
st.caption(f"Model: Informer | Prediction Length: {PRED_LEN} days")

# Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"].tail(250), y=df["Close"].tail(250), name="History"))
fig.add_trace(go.Scatter(x=future_dates, y=pred_prices, name="Prediction", line=dict(dash="dash", color="#FF9100")))
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# Metrics
st.subheader("Results")
c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"${last_price:.2f}")
c2.metric(f"Price in {PRED_LEN} days", f"${pred_prices[-1]:.2f}")
c3.metric("Return", f"{((pred_prices[-1]/last_price)-1)*100:.2f}%")

st.success("✅ Prediction completed successfully!")
