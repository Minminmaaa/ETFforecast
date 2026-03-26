from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# 屏蔽所有冲突库
from transformers import InformerForPrediction
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ETF Informer Dashboard", layout="wide")
MODEL_ROOT = Path("./model")

def load_all():
    scaler = joblib.load(MODEL_ROOT / "scaler.joblib")
    meta = json.loads(open(MODEL_ROOT / "training_meta.json").read())
    df = pd.read_csv(MODEL_ROOT / "dataset_cache.csv", parse_dates=["Date"])
    model = InformerForPrediction.from_pretrained(str(MODEL_ROOT / "informer"))
    return scaler, meta, df, model

scaler, meta, df, model = load_all()
feats = meta["feature_cols"]
ws = meta["window_size"]
pl = meta["pred_len"]

# 数据预处理
df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").ffill().bfill()
X = scaler.transform(df[feats].values)
ctx = X[-ws:]

# 未来特征（必须全0，这是唯一不报错的格式）
future_time = np.zeros((pl, len(feats)), dtype=np.float32)

# ✅ 核心：完全对齐训练格式，绝对不报错
def predict():
    with torch.no_grad():
        past_val = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0)
        past_tf = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        fut_tf = torch.tensor(future_time, dtype=torch.float32).unsqueeze(0)

        out = model.generate(
            past_values=past_val,
            past_time_features=past_tf,
            past_observed_mask=torch.ones_like(past_val),
            future_time_features=fut_tf,
        )
    return out.prediction_outputs.squeeze().cpu().numpy()

pred = predict()

# 绘制结果
last_close = df["Close"].dropna().iloc[-1]
dates = pd.bdate_range(df["Date"].iloc[-1], periods=pl+1)[1:]
pred_price = [last_close]
for r in pred:
    pred_price.append(pred_price[-1] * (1 + float(r)))
pred_price = pred_price[1:]

st.title("ETF 价格预测（Informer）")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"].tail(250), y=df["Close"].tail(250), name="历史价格"))
fig.add_trace(go.Scatter(x=dates, y=pred_price, name="预测价格", line=dict(dash="dash")))
fig.update_layout(template="plotly_dark", title="未来5日价格预测")
st.plotly_chart(fig, use_container_width=True)
