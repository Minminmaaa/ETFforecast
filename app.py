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

# 页面配置
st.set_page_config(page_title="ETF Informer Dashboard", layout="wide")

# 模型路径配置
MODEL_ROOT = Path("./model")
MODEL_CANDIDATE_DIRS = ["informer", "informer_v2", "informer_v3"]

# ----------------------
# 工具函数
# ----------------------
def discover_model_subdirs() -> list[str]:
    found = [d for d in MODEL_CANDIDATE_DIRS if (MODEL_ROOT / d).exists()]
    return found[:3]

def load_df():
    cache_file = MODEL_ROOT / "dataset_cache.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["Date"])
    
    ds = load_dataset("P2SAMAPA/my-etf-data")
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    
    date_col = "Date" if "Date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).rename(columns={date_col: "Date"})
    
    # 缓存到本地，下次更快
    df.to_csv(cache_file, index=False)
    return df

# ----------------------
# 界面主体
# ----------------------
st.title("ETF 时间序列预测 Dashboard")

# 检测模型
model_subdirs = discover_model_subdirs()
if not model_subdirs:
    st.error("未检测到本地模型，请先运行训练脚本！")
    st.stop()

# 加载模型
try:
    models = {
        d: InformerForPrediction.from_pretrained(str(MODEL_ROOT / d))
        for d in model_subdirs
    }
except Exception as e:
    st.error(f"模型加载失败：{str(e)}")
    st.stop()

# 模型选择下拉框
model_options = list(models.keys())
if len(model_options) >= 2:
    model_options.append("ensemble")

model_version = st.sidebar.selectbox("选择模型版本", model_options)

# 加载数据、缩放器、元数据
scaler = joblib.load(MODEL_ROOT / "scaler.joblib")
meta = json.loads((MODEL_ROOT / "training_meta.json").read_text(encoding="utf-8"))
df = load_df()

# 数据清洗
feature_cols = [c for c in meta["feature_cols"] if c in df.columns]
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[feature_cols] = df[feature_cols].ffill().bfill()

# 构造模型输入
x = scaler.transform(df[feature_cols].values)
window_size = int(meta["window_size"])
pred_len = int(meta["pred_len"])
context = x[-window_size:]  # 取最后一段窗口作为输入
future_time_feat = np.repeat(context[-1:, :], pred_len, axis=0)

# ----------------------
# 单模型预测函数
# ----------------------
def run_prediction(model: InformerForPrediction) -> np.ndarray:
    with torch.no_grad():
        # 只取 Close 列作为预测目标（第4列，索引3）
        past_values = torch.tensor(
            context[:, 3:4][None, :, :], dtype=torch.float32
        )
        out = model.generate(
            past_values=past_values,
            past_time_features=torch.tensor(context[None, :, :], dtype=torch.float32),
            past_observed_mask=torch.ones((1, window_size, 1), dtype=torch.float32),
            future_time_features=torch.tensor(future_time_feat[None, :, :], dtype=torch.float32),
        )
    return out.sequences.mean(dim=1).squeeze(0).squeeze(-1).cpu().numpy()

# 执行预测
if model_version == "ensemble":
    predictions = [run_prediction(m) for m in models.values()]
    pred = np.mean(np.stack(predictions), axis=0)
else:
    pred = run_prediction(models[model_version])

# ----------------------
# 构造预测价格
# ----------------------
close_series = pd.to_numeric(df["Close"], errors="coerce").dropna()
last_close = float(close_series.iloc[-1])

# 生成未来日期
last_date = pd.to_datetime(df["Date"].iloc[-1])
future_dates = pd.bdate_range(last_date, periods=pred_len + 1)[1:]

# 收益率 → 价格
pred_price_series = [last_close]
for ret in pred:
    pred_price_series.append(pred_price_series[-1] * (1 + float(ret)))
pred_price_series = np.array(pred_price_series[1:])

# ----------------------
# 绘图
# ----------------------
fig = go.Figure()
# 历史价格（最近250天）
fig.add_trace(
    go.Scatter(
        x=df["Date"].tail(250),
        y=close_series.tail(250),
        name="历史价格",
        line=dict(color="#1f77b4")
    )
)
# 预测价格
fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=pred_price_series,
        name="预测价格",
        line=dict(color="#ff7f0e", dash="dash")
    )
)

fig.update_layout(
    template="plotly_dark",
    title="ETF 价格预测曲线",
    height=500,
    legend=dict(orientation="h", y=1.05)
)

st.plotly_chart(fig, use_container_width=True)

# 展示参数
st.sidebar.markdown("---")
st.sidebar.success(f"""
**模型参数**
窗口大小：{window_size}
预测长度：{pred_len}
当前使用：{model_version}
""")
