from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import InformerForPrediction

st.set_page_config(page_title="ETF Informer Dashboard", layout="wide")
MODEL_ROOT = Path("./model")
MODEL_CANDIDATE_DIRS = ["informer", "informer_v2", "informer_v3"]

def discover_model_subdirs() -> list[str]:
    found = [d for d in MODEL_CANDIDATE_DIRS if (MODEL_ROOT / d).exists()]
    return found[:3]

def load_df():
    cache_file = MODEL_ROOT / "dataset_cache.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["Date"])
    st.warning("使用本地缓存数据")
    return pd.DataFrame()

st.title("ETF Predictive Allocation")
model_subdirs = discover_model_subdirs()

if not model_subdirs:
    st.error("未检测到本地模型，请先运行训练笔记")
else:
    models = {d: InformerForPrediction.from_pretrained(str(MODEL_ROOT / d)) for d in model_subdirs}
    model_options = list(models.keys())
    if len(model_options) >= 2:
        model_options.append("ensemble")
    model_version = st.sidebar.selectbox("模型版本", model_options)

    scaler = joblib.load(MODEL_ROOT / "scaler.joblib")
    meta = json.loads((MODEL_ROOT / "training_meta.json").read_text(encoding="utf-8"))
    df = load_df()

    cols = meta["feature_cols"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cols] = df[cols].ffill().bfill()

    x = scaler.transform(df[cols].values)
    ws = meta["window_size"]
    pl = meta["pred_len"]

    ctx = x[-ws:]
    future_feats = np.zeros((pl, len(cols)), dtype=np.float32)

    def run_one(m: InformerForPrediction) -> np.ndarray:
        with torch.no_grad():
            past_values = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0)
            past_time = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
            future_time = torch.tensor(future_feats, dtype=torch.float32).unsqueeze(0)

            out = m.generate(
                past_values=past_values,
                past_time_features=past_time,
                past_observed_mask=torch.ones_like(past_values),
                future_time_features=future_time,
            )
        return out.prediction_outputs.squeeze().cpu().numpy()

    if model_version == "ensemble":
        preds = [run_one(m) for m in models.values()]
        pred = np.mean(preds, axis=0)
    else:
        pred = run_one(models[model_version])

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    last_close = float(close.iloc[-1])
    future_dates = pd.bdate_range(df["Date"].iloc[-1], periods=pl+1)[1:]

    pred_price = [last_close]
    for r in pred:
        pred_price.append(pred_price[-1] * (1 + float(r)))
    pred_price = pred_price[1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"].tail(250), y=close.tail(250), name="Actual"))
    fig.add_trace(go.Scatter(x=future_dates, y=pred_price, name="Predicted", line=dict(dash="dash")))
    fig.update_layout(template="plotly_dark", title="ETF Price Forecast")
    st.plotly_chart(fig, use_container_width=True)
