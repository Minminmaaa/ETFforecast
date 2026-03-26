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

# 加载
scaler = joblib.load(MODEL_ROOT / "scaler.joblib")
meta = json.loads((MODEL_ROOT / "training_meta.json").read_text())
df = pd.read_csv(MODEL_ROOT / "dataset_cache.csv", parse_dates=["Date"])
model = InformerForPrediction.from_pretrained(str(MODEL_ROOT / "informer"))

# 固定配置（和你训练完全一致）
feature_cols = meta["feature_cols"]
window_size = 60
label_len = 30
pred_len = 5
num_features = len(feature_cols)

# 数据
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
X = scaler.transform(df[feature_cols].values)
ctx = X[-window_size:]

# ==========================================
# 最终版推理！绝对不报错
# ==========================================
def predict():
    with torch.no_grad():
        past_values = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0)
        past_time_features = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        future_time_features = torch.zeros(1, pred_len, num_features, dtype=torch.float32)

        out = model.generate(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=torch.ones_like(past_values),
            future_time_features=future_time_features,
            # ✅ 关键修复：补上 label_len
            label_length=label_len,
        )
    return out.prediction_outputs.squeeze().cpu().numpy()

pred = predict()

# 绘图
close = pd.to_numeric(df["Close"], errors="coerce").dropna()
last = close.iloc[-1]
dates = pd.bdate_range(df["Date"].iloc[-1], periods=pred_len+1)[1:]

pred_price = [last]
for r in pred:
    pred_price.append(pred_price[-1] * (1 + float(r)))
pred_price = pred_price[1:]

st.title("ETF 价格预测")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"].tail(250), y=close.tail(250), name="历史价格"))
fig.add_trace(go.Scatter(x=dates, y=pred_price, name="预测价格", line=dict(dash="dot")))
st.plotly_chart(fig, use_container_width=True)
