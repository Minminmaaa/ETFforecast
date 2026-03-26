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
    st.warning("使用缓存数据")
    return pd.DataFrame()

st.title("ETF Price Forecast (Informer)")
model_subdirs = discover_model_subdirs()

if not model_subdirs:
    st.error("未找到模型，请先运行训练脚本")
else:
    models = {d: InformerForPrediction.from_pretrained(str(MODEL_ROOT / d)) for d in model_subdirs}
    model_options = list(models.keys())
    if len(model_options) >= 2:
        model_options.append("ensemble")
    model_version = st.sidebar.selectbox("模型版本", model_options)

    scaler = joblib.load(MODEL_ROOT / "scaler.joblib")
    meta = json.loads((MODEL_ROOT / "training_meta.json").read_text(encoding="utf-8"))
    df = load_df()

    feature_cols = meta["feature_cols"]
    window_size = meta["window_size"]
    pred_len = meta["pred_len"]

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].ffill().bfill()

    x = scaler.transform(df[feature_cols].values)
    ctx = x[-window_size:]

    # ✅ 关键修复：和训练完全一致 → future 用 0 矩阵
    future_features = np.zeros((pred_len, len(feature_cols)), dtype=np.float32)

    def run_one(m):
        with torch.no_grad():
            past_val = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0)
            past_tf = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
            future_tf = torch.tensor(future_features, dtype=torch.float32).unsqueeze(0)

            out = m.generate(
                past_values=past_val,
                past_time_features=past_tf,
                past_observed_mask=torch.ones_like(past_val),
                future_time_features=future_tf,
            )
        return out.prediction_outputs.cpu().numpy().squeeze()

    if model_version == "ensemble":
        pred = np.mean([run_one(m) for m in models.values()], axis=0)
    else:
        pred = run_one(models[model_version])

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    last_close = float(close.iloc[-1])
    future_dates = pd.bdate_range(df["Date"].iloc[-1], periods=pred_len+1)[1:]

    pred_price = [last_close]
    for r in pred:
        pred_price.append(pred_price[-1] * (1 + float(r)))
    pred_price = pred_price[1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"].tail(250), y=close.tail(250), name="Actual Price"))
    fig.add_trace(go.Scatter(x=future_dates, y=pred_price, name="Predicted", line=dict(dash="dot")))
    fig.update_layout(template="plotly_dark", title="ETF Next 5 Days Prediction")
    st.plotly_chart(fig, use_container_width=True)
