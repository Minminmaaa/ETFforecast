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
    # 本地有缓存就不用 datasets，避免报错
    try:
        from datasets import load_dataset
        ds = load_dataset("P2SAMAPA/my-etf-data")
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        date_col = "Date" if "Date" in df.columns else "date"
        df[date_col] = pd.to_datetime(df[date_col])
        return df.sort_values(date_col).rename(columns={date_col: "Date"})
    except:
        st.warning("数据集加载失败，使用本地缓存")
        return pd.DataFrame()

st.title("ETF Predictive Allocation")
model_subdirs = discover_model_subdirs()
if not model_subdirs:
    st.error("未检测到本地模型，请先运行 notebooks/train.ipynb。")
else:
    models = {d: InformerForPrediction.from_pretrained(str(MODEL_ROOT / d)) for d in model_subdirs}
    model_options = list(models.keys())
    if len(model_options) >= 2:
        model_options.append("ensemble")
    model_version = st.sidebar.selectbox("模型版本", model_options)

    scaler = joblib.load(MODEL_ROOT / "scaler.joblib")
    meta = json.loads((MODEL_ROOT / "training_meta.json").read_text(encoding="utf-8"))
    df = load_df()

    cols = [c for c in meta["feature_cols"] if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cols] = df[cols].ffill().bfill()

    x = scaler.transform(df[cols].values)
    ws = int(meta["window_size"])
    pl = int(meta["pred_len"])
    ctx = x[-ws:]
    future = np.repeat(ctx[-1:, :], pl, axis=0)

    # ✅ 完全修复：numpy 不支持 unsqueeze，先转 tensor 再加维度
    def run_one(m: InformerForPrediction) -> np.ndarray:
        with torch.no_grad():
            past_values = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0)
            past_time_features = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
            future_time_features = torch.tensor(future, dtype=torch.float32).unsqueeze(0)

            out = m.generate(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=torch.ones_like(past_values),
                future_time_features=future_time_features,
            )
        return out.prediction_outputs.squeeze().cpu().numpy()

    if model_version == "ensemble":
        pred = np.mean(np.stack([run_one(m) for m in models.values()], axis=0), axis=0)
    else:
        pred = run_one(models[model_version])

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    last_close = float(close.iloc[-1])
    future_dates = pd.bdate_range(pd.to_datetime(df["Date"].iloc[-1]), periods=pl + 1)[1:]

    pred_price = [last_close]
    for r in pred:
        pred_price.append(pred_price[-1] * (1 + float(r)))
    pred_price = np.array(pred_price[1:])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df["Date"]).tail(250),
        y=close.tail(250),
        name="Actual Price",
        line=dict(color="#1f77b4")
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=pred_price,
        name="Predicted",
        line=dict(color="#ff7f0e", dash="dash")
    ))
    fig.update_layout(
        template="plotly_dark",
        title="ETF Price Forecast (Informer)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
