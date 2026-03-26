from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import InformerForPrediction

from src.pipeline import (
    BASE_FEATURES,
    WindowConfig,
    add_target,
    load_etf_dataframe,
    load_training_meta,
    transform_features,
)


st.set_page_config(page_title="ETF Informer Dashboard", layout="wide")

MODEL_ROOT = Path("./model")
MODEL_CANDIDATE_DIRS = ["informer", "informer_v2", "informer_v3"]


def _detect_symbol_col(df: pd.DataFrame) -> str | None:
    for c in ["Symbol", "symbol", "Ticker", "ticker", "ETF", "etf"]:
        if c in df.columns:
            return c
    return None


@st.cache_resource
def load_model_artifacts(model_dir: Path, model_subdirs: List[str]) -> Tuple[Dict[str, InformerForPrediction], object, Dict]:
    models: Dict[str, InformerForPrediction] = {}
    for subdir in model_subdirs:
        models[subdir] = InformerForPrediction.from_pretrained(str(model_dir / subdir))

    scaler = joblib.load(model_dir / "scaler.joblib")
    meta = load_training_meta(str(model_dir))
    return models, scaler, meta


def discover_model_subdirs(model_root: Path) -> List[str]:
    found: List[str] = []
    for d in MODEL_CANDIDATE_DIRS:
        if (model_root / d).exists():
            found.append(d)
    return found[:3]


@st.cache_data
def load_data() -> pd.DataFrame:
    df = load_etf_dataframe("P2SAMAPA/my-etf-data")
    return add_target(df, pred_len=5)


def build_inference_tensors(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler,
    cfg: WindowConfig,
) -> Dict[str, torch.Tensor]:
    features = transform_features(df, scaler, feature_cols)
    ctx = features[-cfg.window_size :]

    if len(ctx) < cfg.window_size:
        raise ValueError("Not enough rows for selected window size.")

    future_feat = np.repeat(ctx[-1:, :], cfg.pred_len, axis=0)
    past_values = ctx[:, 3:4]

    batch = {
        "past_values": torch.tensor(past_values[None, :, :], dtype=torch.float32),
        "past_time_features": torch.tensor(ctx[None, :, :], dtype=torch.float32),
        "past_observed_mask": torch.ones((1, cfg.window_size, 1), dtype=torch.float32),
        "future_time_features": torch.tensor(future_feat[None, :, :], dtype=torch.float32),
    }
    return batch


def predict_next_returns(
    model: InformerForPrediction,
    infer_inputs: Dict[str, torch.Tensor],
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model.generate(
            past_values=infer_inputs["past_values"],
            past_time_features=infer_inputs["past_time_features"],
            past_observed_mask=infer_inputs["past_observed_mask"],
            future_time_features=infer_inputs["future_time_features"],
        )
    pred = out.sequences.mean(dim=1).squeeze(0).squeeze(-1).cpu().numpy()
    return pred


def predict_with_selected_models(
    models: Dict[str, InformerForPrediction],
    model_version: str,
    infer_inputs: Dict[str, torch.Tensor],
) -> np.ndarray:
    if model_version == "ensemble":
        preds = [predict_next_returns(m, infer_inputs) for m in models.values()]
        return np.mean(np.stack(preds, axis=0), axis=0)

    return predict_next_returns(models[model_version], infer_inputs)


def suggest_weights(symbol_to_pred: Dict[str, float]) -> pd.DataFrame:
    s = pd.Series(symbol_to_pred)
    s = s.clip(lower=0)
    if float(s.sum()) == 0.0:
        s = pd.Series(1.0, index=s.index)
    w = s / s.sum()
    out = w.sort_values(ascending=False).reset_index()
    out.columns = ["ETF", "Weight"]
    return out


def main() -> None:
    st.title("ETF Predictive Allocation")

    model_subdirs = discover_model_subdirs(MODEL_ROOT)
    if not (model_subdirs and (MODEL_ROOT / "scaler.joblib").exists() and (MODEL_ROOT / "training_meta.json").exists()):
        st.error("未检测到本地模型文件，请先运行 notebooks/train.ipynb 完成训练并保存模型。")
        return

    models, scaler, meta = load_model_artifacts(MODEL_ROOT, model_subdirs)
    feature_cols = meta.get("feature_cols", BASE_FEATURES)
    cfg = WindowConfig(
        window_size=int(meta.get("window_size", 60)),
        label_len=int(meta.get("label_len", 30)),
        pred_len=int(meta.get("pred_len", 5)),
    )

    df = load_data()
    symbol_col = _detect_symbol_col(df)

    with st.sidebar:
        st.header("配置")
        if symbol_col:
            symbols = sorted(df[symbol_col].dropna().astype(str).unique().tolist())
        else:
            symbols = ["ALL"]

        selected_symbol = st.selectbox("ETF 标的", symbols)
        horizon = st.slider("预测步长(天)", min_value=1, max_value=7, value=5, step=1)
        model_options = list(models.keys())
        if len(model_options) >= 2:
            model_options.append("ensemble")
        model_version = st.selectbox("模型版本", model_options)

    if symbol_col and selected_symbol != "ALL":
        view_df = df[df[symbol_col].astype(str) == selected_symbol].copy()
    else:
        view_df = df.copy()

    view_df = view_df.sort_values("Date").reset_index(drop=True)

    infer_inputs = build_inference_tensors(view_df, feature_cols, scaler, cfg)
    pred = predict_with_selected_models(models, model_version, infer_inputs)
    pred = pred[:horizon]

    last_close = float(view_df["Close"].iloc[-1])
    yday_close = float(view_df["Close"].iloc[-2]) if len(view_df) > 1 else last_close
    daily_change = (last_close / yday_close - 1.0) * 100
    trend = "Up" if float(np.mean(pred)) >= 0 else "Down"

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"{last_close:.2f}")
    c2.metric("Yesterday Change", f"{daily_change:.2f}%")
    c3.metric("Model Trend", trend)

    history_days = min(250, len(view_df))
    hist = view_df.tail(history_days).copy()
    last_date = pd.to_datetime(hist["Date"].iloc[-1])
    future_dates = pd.bdate_range(last_date, periods=horizon + 1)[1:]

    pred_price = [last_close]
    for r in pred:
        pred_price.append(pred_price[-1] * (1.0 + float(r)))
    pred_price = np.array(pred_price[1:])

    vol = float(view_df["Close"].pct_change().dropna().tail(60).std())
    upper = pred_price * (1.0 + 1.96 * vol)
    lower = pred_price * (1.0 - 1.96 * vol)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=hist["Date"], y=hist["Close"], name="Actual", mode="lines"))
    fig_price.add_trace(go.Scatter(x=future_dates, y=pred_price, name="Predicted", mode="lines"))
    fig_price.add_trace(go.Scatter(x=future_dates, y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_price.add_trace(go.Scatter(x=future_dates, y=lower, fill="tonexty", name="Prediction Interval", line=dict(width=0), hoverinfo="skip"))
    fig_price.update_layout(template="plotly_dark", title="Historical Price and Prediction Band")
    st.plotly_chart(fig_price, use_container_width=True)

    if symbol_col:
        corr_src = (
            df.pivot_table(index="Date", columns=symbol_col, values="Close", aggfunc="last")
            .pct_change()
            .dropna(how="all")
        )
        if corr_src.shape[1] >= 2:
            corr = corr_src.corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="ETF Correlation Heatmap")
            fig_corr.update_layout(template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)

    if symbol_col:
        symbols = sorted(df[symbol_col].dropna().astype(str).unique().tolist())
    else:
        symbols = ["ALL"]

    pred_map: Dict[str, float] = {}
    for sym in symbols:
        if symbol_col and sym != "ALL":
            sym_df = df[df[symbol_col].astype(str) == sym].sort_values("Date").reset_index(drop=True)
        else:
            sym_df = df.sort_values("Date").reset_index(drop=True)

        if len(sym_df) < cfg.window_size + cfg.pred_len:
            continue

        try:
            inputs = build_inference_tensors(sym_df, feature_cols, scaler, cfg)
            p = predict_with_selected_models(models, model_version, inputs)
            pred_map[sym] = float(np.mean(p[:horizon]))
        except Exception:
            continue

    if pred_map:
        weights = suggest_weights(pred_map)
        fig_w = px.bar(weights, x="ETF", y="Weight", title="Suggested Portfolio Weights")
        fig_w.update_layout(template="plotly_dark", yaxis_tickformat=".0%")
        st.plotly_chart(fig_w, use_container_width=True)

    st.caption(f"Model: {model_version} | Data: P2SAMAPA/my-etf-data")


if __name__ == "__main__":
    main()
