import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from datetime import datetime, timedelta
from transformers import InformerConfig, InformerForPrediction
from utils import (load_etf_dataframe, add_target, split_train_val_test,
                   WindowConfig, InformerWindowDataset, InformerDataCollator,
                   BASE_FEATURES)

# 设置页面
st.set_page_config(page_title="ETF Informer", layout="wide")
st.title("📈 ETF 价格预测 - Informer 模型")

# 加载模型和 scaler
@st.cache_resource
def load_models_and_scaler(model_dir="model"):
    model_dir = Path(model_dir)
    scaler = joblib.load(model_dir / "scaler.joblib")
    
    models = {}
    configs = {}
    for name in ["informer", "informer_v2", "informer_v3"]:
        config = InformerConfig.from_pretrained(str(model_dir / name))
        model = InformerForPrediction.from_pretrained(str(model_dir / name))
        model.eval()
        models[name] = model
        configs[name] = config
    return scaler, models, configs

# 数据加载（缓存）
@st.cache_data
def load_data():
    df = load_etf_dataframe()  # 确保这个函数能获取数据
    # 生成目标列（假设 pred_len = 5）
    df = add_target(df, pred_len=5)
    return df

# 预测函数
def predict_returns(model, scaler, context_df, pred_len, device):
    """
    context_df: 包含历史的 DataFrame（按时间排序），至少包含 window_size 行
    返回预测的收益率序列 (pred_len,)
    """
    # 准备输入
    feature_cols = BASE_FEATURES
    X = scaler.transform(context_df[feature_cols].values)
    ctx = X[-window_cfg.window_size:]  # 取最后窗口
    # 未来特征用最后一个时间步重复
    future_feat = np.repeat(ctx[-1:], pred_len, axis=0)
    
    # 转为 tensor
    past_values = torch.tensor(ctx[:, 3:4], dtype=torch.float32).unsqueeze(0)  # (1, T, 1)
    past_time = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
    past_mask = torch.ones_like(past_values)
    future_time = torch.tensor(future_feat, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        # 尝试 generate，否则 fallback
        try:
            output = model.generate(
                past_values=past_values,
                past_time_features=past_time,
                past_observed_mask=past_mask,
                future_time_features=future_time,
            )
            pred = output.sequences.mean(dim=1).squeeze().cpu().numpy()
        except RuntimeError:
            # fallback: 用 forward 并取 mean
            future_values = torch.zeros((1, pred_len, 1), dtype=torch.float32)
            out = model(
                past_values=past_values,
                past_time_features=past_time,
                past_observed_mask=past_mask,
                future_values=future_values,
                future_time_features=future_time,
            )
            pred = out.params[1].squeeze().cpu().numpy()
    return pred[:pred_len]

# 主应用
def main():
    # 侧边栏配置
    st.sidebar.header("模型配置")
    model_name = st.sidebar.selectbox("选择模型", ["informer", "informer_v2", "informer_v3"])
    window_size = st.sidebar.slider("窗口大小（历史天数）", 30, 120, 60)
    pred_len = st.sidebar.slider("预测天数", 1, 20, 5)
    
    # 加载资源
    with st.spinner("加载模型和数据..."):
        scaler, models, configs = load_models_and_scaler()
        df = load_data()
    
    # 显示数据概况
    st.subheader("数据概览")
    st.write(f"数据时间范围: {df['Date'].min()} 至 {df['Date'].max()}")
    st.write(f"总样本数: {len(df)}")
    
    # 交互预测区域
    st.subheader("实时预测")
    col1, col2 = st.columns(2)
    with col1:
        # 日期选择器
        min_date = pd.to_datetime(df['Date']).min()
        max_date = pd.to_datetime(df['Date']).max()
        start_date = st.date_input("参考日期（预测起始点）", 
                                   value=pd.to_datetime(df['Date']).max() - timedelta(days=30),
                                   min_value=min_date, max_value=max_date)
        pred_len_input = st.number_input("预测天数", min_value=1, max_value=20, value=pred_len)
    
    if st.button("生成预测"):
        with st.spinner("预测中..."):
            # 提取参考日期之前的数据
            selected_date = pd.to_datetime(start_date)
            hist_df = df[pd.to_datetime(df['Date']) <= selected_date].reset_index(drop=True)
            if len(hist_df) < window_size:
                st.error(f"历史数据不足 {window_size} 天，请选择更晚的日期或减小窗口大小")
                return
            
            # 获取模型和配置
            model = models[model_name]
            config = configs[model_name]
            # 使用用户选择的预测长度（但模型训练时固定为5，这里需要适配）
            # 注意：如果模型训练时 pred_len=5，预测超过5天会出错，需要截断
            if pred_len_input > config.prediction_length:
                st.warning(f"模型最大预测长度为 {config.prediction_length}，将截断为 {config.prediction_length} 天")
                pred_len_use = config.prediction_length
            else:
                pred_len_use = pred_len_input
            
            # 进行预测
            pred_returns = predict_returns(model, scaler, hist_df, pred_len_use, device='cpu')
            
            # 转换为价格
            last_close = hist_df['Close'].iloc[-1]
            pred_prices = [last_close]
            for r in pred_returns:
                pred_prices.append(pred_prices[-1] * (1 + r))
            pred_prices = np.array(pred_prices[1:])
            
            # 构建未来日期
            last_date = selected_date
            future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=pred_len_use)
            
            # 绘图
            fig = go.Figure()
            # 历史价格（最近60天）
            hist_plot = hist_df.tail(window_size).copy()
            fig.add_trace(go.Scatter(x=hist_plot['Date'], y=hist_plot['Close'],
                                     name='历史价格', line=dict(color='steelblue')))
            fig.add_trace(go.Scatter(x=future_dates, y=pred_prices,
                                     name='预测价格', mode='lines+markers',
                                     line=dict(color='orange', dash='dash'),
                                     marker=dict(size=8)))
            # 标记参考点
            fig.add_vline(x=selected_date, line=dict(color='red', dash='dot'))
            fig.add_annotation(x=selected_date, y=1, yref='paper', text='参考点',
                               showarrow=False, xanchor='left', yshift=10)
            fig.update_layout(title=f"{model_name} 预测",
                              xaxis_title="日期", yaxis_title="收盘价",
                              template="plotly_dark", hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # 结果显示
            st.subheader("预测摘要")
            st.write(f"参考日期: {selected_date.date()}")
            st.write(f"参考收盘价: {last_close:.2f}")
            st.write(f"预测 {pred_len_use} 天后价格: {pred_prices[-1]:.2f}")
            st.write(f"预期收益率: {(pred_prices[-1]/last_close - 1)*100:.2f}%")
            
            # 显示预测表格
            pred_df = pd.DataFrame({
                "日期": future_dates,
                "预测收益率": pred_returns,
                "预测收盘价": pred_prices
            })
            st.dataframe(pred_df)
    
    # 特征相关性（可选）
    st.subheader("特征相关性分析")
    corr = df[BASE_FEATURES].corr()
    fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                         title="特征相关性热图")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # 误差分布（如果测试集已计算）
    # 可以在代码中预先计算测试集预测，但为了简洁，跳过

if __name__ == "__main__":
    main()