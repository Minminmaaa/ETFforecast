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
# 页面配置（公开环境安全）
# ------------------------------
st.set_page_config(
    page_title="ETF Price Predictor",
    layout="wide",
)

# ------------------------------
# 模型路径（公开环境标准路径）
# ------------------------------
MODEL_ROOT = Path("./model")

@st.cache_resource(show_spinner="Loading AI Model...")
def load_model_and_data():
    model = InformerForPrediction.from_pretrained(str(MODEL_ROOT / "informer"))
    scaler = joblib.load(MODEL_ROOT / "scaler.joblib")
    meta = json.load(open(MODEL_ROOT / "training_meta.json"))
    df = pd.read_csv(MODEL_ROOT / "dataset_cache.csv", parse_dates=["Date"])
    return model, scaler, meta, df

# 加载失败则停止
try:
    model, scaler, meta, df = load_model_and_data()
except Exception as e:
    st.error(f"❌ Model load failed: {str(e)}")
    st.stop()

# ------------------------------
# 配置
# ------------------------------
FEATURES = meta["feature_cols"]
WINDOW = meta["window_size"]
HORIZON = meta["pred_len"]

# 数据清洗
df = df.sort_values("Date").reset_index(drop=True)
df[FEATURES] = df[FEATURES].astype(float).ffill().fillna(0)

# ------------------------------
# ✅ 【最稳定】推理函数（无任何报错）
# ------------------------------
def get_prediction():
    X = scaler.transform(df[FEATURES].values)
    ctx = X[-WINDOW:]

    past_val = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0)
    past_tf = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
    future_tf = torch.zeros((1, HORIZON, len(FEATURES)), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model.generate(
            past_values=past_val,
            past_time_features=past_tf,
            future_time_features=future_tf,
            past_observed_mask=torch.ones_like(past_val),
        )
    
    pred = output.sequences.mean(dim=1).squeeze().cpu().numpy()
    return pred if pred.ndim > 0 else [pred]

# ------------------------------
# 推理 & 计算价格
# ------------------------------
pred_returns = get_prediction()
last_price = df["Close"].iloc[-1]
last_dt = df["Date"].iloc[-1]

# 预测价格
future_dates = pd.bdate_range(last_dt, periods=HORIZON+1, freq="B")[1:]
pred_prices = [last_price]
for r in pred_returns:
    pred_prices.append(pred_prices[-1] * (1 + float(r)))

pred_final = pred_prices[-1]
ret = ((pred_final / last_price) - 1) * 100

# ------------------------------
# 界面展示（公开可用）
# ------------------------------
st.title("📈 ETF Price Prediction Dashboard")
st.caption(f"✅ Informer Model | {HORIZON} Days Forecast")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.Date.tail(250), y=df.Close.tail(250), name="History"))
fig.add_trace(go.Scatter(x=future_dates, y=pred_prices[1:], name="Prediction", line=dict(dash="dash")))
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# 结果卡片
st.subheader("Prediction Result")
c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"${last_price:.2f}")
c2.metric(f"After {HORIZON} Days", f"${pred_final:.2f}")
c3.metric("Expected Return", f"{ret:.2f}%")

st.success("✅ Prediction completed!")
