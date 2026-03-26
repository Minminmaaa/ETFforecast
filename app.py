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

st.set_page_config(page_title="ETF Forecast", layout="wide")
MODEL_ROOT = Path("./model")

# 加载所有必需文件
scaler = joblib.load(MODEL_ROOT / "scaler.joblib")
meta = json.loads((MODEL_ROOT / "training_meta.json").read_text())
df = pd.read_csv(MODEL_ROOT / "dataset_cache.csv", parse_dates=["Date"])
model = InformerForPrediction.from_pretrained(str(MODEL_ROOT / "informer"))

# 配置（完全对齐训练）
feature_cols = meta["feature_cols"]
window_size = meta["window_size"]
pred_len = meta["pred_len"]
num_features = len(feature_cols)

# 数据预处理
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
df[feature_cols] = df[feature_cols].ffill().bfill()
X = scaler.transform(df[feature_cols].values)
ctx = X[-window_size:]

# ===================== 核心修复 =====================
# 完全按照 HuggingFace 官方格式输入
# ====================================================
def predict():
    with torch.no_grad():
        past_values        = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0)  # [1, 60, 1]
        past_time_features = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)          # [1, 60, 9]
        future_time_features = torch.zeros((1, pred_len, num_features), dtype=torch.float32)

        out = model.generate(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=torch.ones_like(past_values),
            future_time_features=future_time_features,
        )
    return out.prediction_outputs.squeeze().cpu().numpy()

# 预测
pred = predict()

# 计算预测价格
close_series = pd.to_numeric(df["Close"], errors="coerce").dropna()
last_close = float(close_series.iloc[-1])
future_dates = pd.bdate_range(df["Date"].iloc[-1], periods=pred_len + 1)[1:]

pred_price = [last_close]
for ret in pred:
    pred_price.append(pred_price[-1] * (1 + float(ret)))
pred_price = pred_price[1:]

# 绘图
st.title("ETF Price Prediction (Informer)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"].tail(250), y=close_series.tail(250), name="Actual"))
fig.add_trace(go.Scatter(x=future_dates, y=pred_price, name="Predicted", line=dict(dash="dash")))
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)
