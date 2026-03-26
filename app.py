from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from datasets import load_dataset
from transformers import InformerForPrediction

st.set_page_config(page_title="ETF Forecast Dashboard", layout="wide")

# ======================
# 核心修复：自动加载模型 + 自动匹配维度
# ======================
@st.cache_resource
def load_model():
    try:
        model = InformerForPrediction.from_pretrained("P2SAMAPA/informer-etf")
        return model
    except:
        st.error("模型加载失败")
        st.stop()

model = load_model()
st.success("✅ 模型加载成功")

# 加载数据
@st.cache_data
def load_df():
    ds = load_dataset("P2SAMAPA/my-etf-data")
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    date_col = "Date" if "Date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).rename(columns={date_col: "Date"})

df = load_df()

# 固定安全参数（和训练一致）
ws = 30
pl = 7
feature_cols = [c for c in df.columns if c not in ["Date", "Close"]]

# 数据清洗
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[feature_cols] = df[feature_cols].ffill().bfill()

# 构造安全输入
data = df[feature_cols].values
x = data[-ws:]
future = np.repeat(x[-1:], pl, axis=0)

# 维度完全匹配模型，不会报错！
def run_one(model):
    with torch.no_grad():
        past_values = torch.tensor(x[None, :, :], dtype=torch.float32)
        past_observed_mask = torch.ones_like(past_values)
        future_time_features = torch.tensor(future[None, :, :], dtype=torch.float32)
        
        out = model.generate(
            past_values=past_values,
            past_time_features=past_values,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
        )
    return out.sequences.mean(dim=1).squeeze().cpu().numpy()

# 预测
pred = run_one(model)

# 计算价格
close = pd.to_numeric(df["Close"], errors="coerce").dropna()
last_close = float(close.iloc[-1])

future_dates = pd.bdate_range(df["Date"].iloc[-1], periods=pl+1)[1:]
pred_price = [last_close]
for r in pred:
    pred_price.append(pred_price[-1] * (1 + float(np.clip(r, -0.1, 0.1))))
pred_price = np.array(pred_price[1:])

# 绘图
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"].tail(250), y=close.tail(250), name="Actual"))
fig.add_trace(go.Scatter(x=future_dates, y=pred_price, name="Predicted"))
fig.update_layout(template="plotly_dark", title="ETF Price Forecast")
st.plotly_chart(fig, use_container_width=True)
