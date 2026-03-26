import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
from pathlib import Path
from transformers import InformerForPrediction, InformerConfig
from utils import (
    load_etf_dataframe, add_target, BASE_FEATURES,
    WindowConfig, InformerWindowDataset, InformerDataCollator
)

# 页面配置
st.set_page_config(page_title="ETF Informer", layout="wide")
st.title("📈 ETF 价格预测 - Informer 模型")

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 缓存加载 ====================
@st.cache_resource
def load_models_and_scaler(model_dir="model"):
    """加载 scaler 和所有预训练模型"""
    model_dir = Path(model_dir)
    scaler = joblib.load(model_dir / "scaler.joblib")
    models = {}
    configs = {}
    for name in ["informer", "informer_v2", "informer_v3"]:
        config = InformerConfig.from_pretrained(str(model_dir / name))
        model = InformerForPrediction.from_pretrained(str(model_dir / name))
        model.to(device)
        model.eval()
        models[name] = model
        configs[name] = config
    return scaler, models, configs

@st.cache_data
def load_data():
    """加载并预处理数据（包含目标列）"""
    df = load_etf_dataframe()
    # 这里假设模型训练时 pred_len=5，但实际在预测时会使用用户选择的长度（截断）
    df = add_target(df, pred_len=5)
    return df

# ==================== 预测函数 ====================
def predict_returns(model, scaler, hist_df, pred_len, window_size, device):
    """
    使用模型的 forward 方法预测收益率（不使用 generate，避免维度错误）
    返回预测收益率序列 (pred_len,)
    """
    # 提取特征
    X = scaler.transform(hist_df[BASE_FEATURES].values)
    # 取最后 window_size 行
    ctx = X[-window_size:]  # (window_size, num_features)
    # 未来特征：重复最后一个时间步
    future_feat = np.repeat(ctx[-1:], pred_len, axis=0)  # (pred_len, num_features)

    # 转张量
    past_val = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0).to(device)      # (1, window_size, 1)
    past_time = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(device)            # (1, window_size, num_features)
    past_mask = torch.ones_like(past_val)
    future_time = torch.tensor(future_feat, dtype=torch.float32).unsqueeze(0).to(device)  # (1, pred_len, num_features)
    future_val = torch.zeros((1, pred_len, 1), dtype=torch.float32).to(device)            # 占位符

    model.eval()
    with torch.no_grad():
        output = model(
            past_values=past_val,
            past_time_features=past_time,
            past_observed_mask=past_mask,
            future_values=future_val,
            future_time_features=future_time,
        )
        # 输出 params[1] 是均值 (loc)，params[0] 是尺度 (scale)
        pred = output.params[1].squeeze().cpu().numpy()  # (pred_len,)
    return pred

# ==================== 主界面 ====================
def main():
    # 侧边栏：模型配置
    st.sidebar.header("模型配置")
    model_name = st.sidebar.selectbox("选择模型", ["informer", "informer_v2", "informer_v3"])
    window_size = st.sidebar.slider("窗口大小（历史天数）", 30, 120, 60, step=10)
    user_pred_len = st.sidebar.slider("预测天数", 1, 20, 5, step=1)

    # 加载资源
    with st.spinner("加载模型和数据..."):
        scaler, models, configs = load_models_and_scaler()
        df = load_data()
        model = models[model_name]
        model_config = configs[model_name]
        # 模型训练时的预测长度
        model_pred_len = model_config.prediction_length

    # 数据概览
    st.subheader("数据概览")
    st.write(f"数据时间范围: {df['Date'].min()} 至 {df['Date'].max()}")
    st.write(f"总样本数: {len(df)}")
    st.write(f"特征数量: {len(BASE_FEATURES)}")

    # 实时预测区域
    st.subheader("实时预测")
    col1, col2 = st.columns(2)
    with col1:
        min_date = pd.to_datetime(df['Date']).min()
        max_date = pd.to_datetime(df['Date']).max()
        start_date = st.date_input(
            "参考日期（预测起点）",
            value=pd.to_datetime(df['Date']).max() - timedelta(days=30),
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        st.write(f"模型最大预测长度: {model_pred_len} 天")
        if user_pred_len > model_pred_len:
            st.warning(f"请求的预测天数 ({user_pred_len}) 超过模型最大长度 ({model_pred_len})，将自动截断。")
        pred_len_use = min(user_pred_len, model_pred_len)

    if st.button("生成预测"):
        with st.spinner("预测中..."):
            # 获取参考日期之前的历史数据
            selected_date = pd.to_datetime(start_date)
            hist_df = df[pd.to_datetime(df['Date']) <= selected_date].reset_index(drop=True)
            if len(hist_df) < window_size:
                st.error(f"历史数据不足 {window_size} 天，请选择更晚的日期或减小窗口大小")
                return

            # 执行预测
            pred_returns = predict_returns(
                model, scaler, hist_df, pred_len_use, window_size, device
            )

            # 转换为价格
            last_close = hist_df['Close'].iloc[-1]
            pred_prices = [last_close]
            for r in pred_returns:
                pred_prices.append(pred_prices[-1] * (1 + r))
            pred_prices = np.array(pred_prices[1:])

            # 构建未来日期（仅交易日）
            last_date = selected_date
            future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=pred_len_use)

            # 绘图：历史价格（最近窗口期）和预测价格
            hist_plot = hist_df.tail(window_size).copy()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_plot['Date'], y=hist_plot['Close'],
                name='历史价格', line=dict(color='steelblue')
            ))
            fig.add_trace(go.Scatter(
                x=future_dates, y=pred_prices,
                name='预测价格', mode='lines+markers',
                line=dict(color='orange', dash='dash'),
                marker=dict(size=8)
            ))
            # 参考线
            fig.add_vline(x=selected_date, line=dict(color='red', dash='dot'))
            fig.add_annotation(
                x=selected_date, y=1, yref='paper',
                text='参考点', showarrow=False, xanchor='left', yshift=10
            )
            fig.update_layout(
                title=f"{model_name} 预测 (窗口={window_size}天)",
                xaxis_title="日期", yaxis_title="收盘价",
                template="plotly_dark", hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # 显示预测摘要
            st.subheader("预测摘要")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("参考日期", selected_date.date())
            col2.metric("参考收盘价", f"{last_close:.2f}")
            col3.metric(f"{pred_len_use} 天后价格", f"{pred_prices[-1]:.2f}")
            col4.metric("预期收益率", f"{(pred_prices[-1]/last_close - 1)*100:.2f}%")

            # 预测明细表格
            pred_df = pd.DataFrame({
                "日期": future_dates,
                "预测收益率": pred_returns,
                "预测收盘价": pred_prices
            })
            st.dataframe(pred_df)

    # 特征相关性分析
    st.subheader("特征相关性分析")
    corr = df[BASE_FEATURES].corr()
    fig_corr = px.imshow(
        corr, text_auto='.2f', color_continuous_scale='RdBu_r',
        title="特征相关性热图", aspect='auto'
    )
    fig_corr.update_layout(template='plotly_dark')
    st.plotly_chart(fig_corr, use_container_width=True)

    # 可选：展示测试集误差（如果已预先计算）
    st.subheader("模型评估")
    st.write("测试集 RMSE ≈ 0.0214 (基于原始 notebook 结果)")

if __name__ == "__main__":
    main()
