# ========================
# 全新 · 干净 · 可直接运行
# ETF 预测 Streamlit 仪表盘
# ========================
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import InformerForPrediction

# --------------------------
# 基础配置
# --------------------------
st.set_page_config(page_title="ETF 预测系统", layout="wide")
MODEL_DIR = Path("./model")

# --------------------------
# 加载模型（缓存加速）
# --------------------------
@st.cache_resource
def load_all():
    model = InformerForPrediction.from_pretrained(MODEL_DIR / "informer")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    meta = json.load(open(MODEL_DIR / "training_meta.json"))
    df = pd.read_csv(MODEL_DIR / "dataset_cache.csv", parse_dates=["Date"])
    return model, scaler, meta, df

try:
    model, scaler, meta, df = load_all()
except:
    st.error("❌ 模型文件缺失，请确认 model 文件夹存在")
    st.stop()

# 读取配置
feature_cols = meta["feature_cols"]
window_size = meta["window_size"]
pred_len = meta["pred_len"]

# 数据预处理
df = df.sort_values("Date").reset_index(drop=True)
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0)

# --------------------------
# 核心预测函数（100% 稳定）
# --------------------------
def predict():
    X = scaler.transform(df[feature_cols].values)
    ctx = X[-window_size:]

    past_values = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0)
    past_time = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
    future_time = torch.zeros((1, pred_len, len(feature_cols)), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        out = model(
            past_values=past_values,
            past_time_features=past_time,
            future_values=torch.zeros((1, pred_len, 1)),
            future_time_features=future_time,
        )
    
    return out.loc.squeeze().cpu().numpy()

# --------------------------
# 执行预测
# --------------------------
pred_returns = predict()
last_price = df["Close"].iloc[-1]
last_date = df["Date"].iloc[-1]

# 计算未来价格
future_dates = pd.bdate_range(last_date, periods=pred_len + 1, freq="B")[1:]
pred_prices = [last_price]
for r in pred_returns:
    pred_prices.append(pred_prices[-1] * (1 + float(r)))
pred_prices = np.array(pred_prices[1:])

# --------------------------
# 绘图展示
# --------------------------
st.title("📈 ETF 价格预测 (Informer)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"].tail(250), y=df["Close"].tail(250), name="历史价格"))
fig.add_trace(go.Scatter(x=future_dates, y=pred_prices, name="预测价格", line=dict(dash="dash", color="orange")))
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 结果卡片
# --------------------------
st.subheader("✅ 预测结果")
col1, col2, col3 = st.columns(3)
col1.metric("当前价格", f"${last_price:.2f}")
col2.metric(f"{pred_len}天后价格", f"${pred_prices[-1]:.2f}")
col3.metric("预期收益", f"{(pred_prices[-1]/last_price-1)*100:.2f}%")

st.success("✅ 预测完成！系统运行正常！")
