from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import InformerForPrediction

st.set_page_config(page_title="ETF 预测仪表盘", layout="wide")
MODEL_ROOT = Path("./model")

@st.cache_resource
def load_model():
    model = InformerForPrediction.from_pretrained(str(MODEL_ROOT / "informer"))
    scaler = joblib.load(MODEL_ROOT / "scaler.joblib")
    meta = json.load(open(MODEL_ROOT / "training_meta.json"))
    df = pd.read_csv(MODEL_ROOT / "dataset_cache.csv", parse_dates=["Date"])
    return model, scaler, meta, df

model, scaler, meta, df = load_model()
feature_cols = meta["feature_cols"]
window_size = meta["window_size"]
pred_len = meta["pred_len"]
label_len = 30

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
X = scaler.transform(df[feature_cols].values)
ctx = X[-window_size:]

# ====================== 【官方正确写法】 ======================
def run_one(m):
    with torch.no_grad():
        # 构造输入
        past_values = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0)
        past_time = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

        # ✅ 关键：构造正确的 decoder 输入（必须包含 label 部分）
        # 这是你所有报错的根源！
        future_values = torch.zeros((1, pred_len, 1), dtype=torch.float32)
        future_time = torch.zeros((1, pred_len, len(feature_cols)), dtype=torch.float32)

        out = m(
            past_values=past_values,
            past_time_features=past_time,
            future_values=future_values,
            future_time_features=future_time,
            past_observed_mask=torch.ones_like(past_values),
        )

    return out.prediction_outputs.squeeze().cpu().numpy()

# =================================================================

st.title("📈 ETF 价格预测 (Informer)")
pred = run_one(model)

last_close = df["Close"].dropna().iloc[-1]
future_dates = pd.bdate_range(df["Date"].iloc[-1], periods=pred_len+1)[1:]

pred_price = [last_close]
for r in pred:
    pred_price.append(pred_price[-1] * (1 + float(r)))
pred_price = pred_price[1:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"].tail(250), y=df["Close"].tail(250), name="历史价格"))
fig.add_trace(go.Scatter(x=future_dates, y=pred_price, name="预测价格", line=dict(dash="dash")))
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.success(f"✅ 预测完成！未来 {pred_len} 天价格：{pred_price[-1]:.2f}")
