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


st.set_page_config(page_title="ETF Informer Dashboard", layout="wide")
MODEL_ROOT_CANDIDATES = [Path("./model"), Path("./notebooks/model")]
MODEL_CANDIDATE_DIRS = ["informer", "informer_v2", "informer_v3"]
BASE_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "CPI",
    "Unemployment Rate",
    "DXY",
    "Gold/Copper Ratio",
]


def detect_model_root() -> Path | None:
    for root in MODEL_ROOT_CANDIDATES:
        if root.exists() and (root / "scaler.joblib").exists() and (root / "training_meta.json").exists():
            return root
    return None


def discover_model_subdirs(model_root: Path) -> list[str]:
    selected: list[str] = []
    for base in MODEL_CANDIDATE_DIRS:
        candidates = []
        for p in model_root.iterdir():
            if not p.is_dir():
                continue
            if p.name == base or p.name.startswith(base + "_backup_"):
                if (p / "config.json").exists():
                    candidates.append(p)
        if candidates:
            candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            selected.append(candidates[0].name)
    return selected[:3]


def extend_pred_to_horizon(pred: np.ndarray, horizon: int) -> np.ndarray:
    pred = np.asarray(pred).reshape(-1)
    if len(pred) >= horizon:
        return pred[:horizon]
    if len(pred) == 0:
        return np.zeros(horizon, dtype=float)
    tail = pred[-min(3, len(pred)) :]
    fill = float(np.mean(tail))
    extra = np.repeat(fill, horizon - len(pred))
    return np.concatenate([pred, extra])


@st.cache_data
def load_df(model_root: Path) -> pd.DataFrame:
    cache_file = model_root / "dataset_cache.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["Date"])

    try:
        ds = load_dataset("P2SAMAPA/my-etf-data")
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        date_col = "Date" if "Date" in df.columns else "date"
        df[date_col] = pd.to_datetime(df[date_col])
        return df.sort_values(date_col).rename(columns={date_col: "Date"})
    except Exception:
        dates = pd.bdate_range("2018-01-01", periods=2000)
        n = len(dates)
        rng = np.random.default_rng(42)
        returns = 0.0002 + 0.01 * rng.standard_normal(n)
        close = 100 * np.exp(np.cumsum(returns))
        open_ = close * (1 + 0.001 * rng.standard_normal(n))
        high = np.maximum(open_, close) * (1 + np.abs(0.003 * rng.standard_normal(n)))
        low = np.minimum(open_, close) * (1 - np.abs(0.003 * rng.standard_normal(n)))
        volume = rng.integers(2_000_000, 10_000_000, n).astype(float)
        return pd.DataFrame(
            {
                "Date": dates,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
                "CPI": 250 + np.cumsum(0.02 + 0.05 * rng.standard_normal(n)),
                "Unemployment Rate": 5 + 0.5 * np.sin(np.linspace(0, 8 * np.pi, n)) + 0.2 * rng.standard_normal(n),
                "DXY": 95 + np.cumsum(0.02 * rng.standard_normal(n)),
                "Gold/Copper Ratio": 0.2 + 0.02 * np.sin(np.linspace(0, 10 * np.pi, n)),
            }
        )


def baseline_predict_returns(df: pd.DataFrame, horizon: int) -> np.ndarray:
    pct = pd.to_numeric(df["Close"], errors="coerce").pct_change().dropna()
    if len(pct) == 0:
        return np.zeros(horizon, dtype=float)
    mu = float(pct.tail(20).mean())
    return np.repeat(mu, horizon).astype(float)


def build_inference_tensors(
    scaled_features: np.ndarray,
    window_size: int,
    pred_len: int,
) -> dict[str, torch.Tensor]:
    ctx = scaled_features[-window_size:]
    future = np.repeat(ctx[-1:, :], pred_len, axis=0)
    return {
        "past_values": torch.tensor(ctx[:, 3][None, :], dtype=torch.float32),
        "past_time_features": torch.tensor(ctx[None, :, :], dtype=torch.float32),
        "past_observed_mask": torch.ones((1, window_size), dtype=torch.float32),
        "future_time_features": torch.tensor(future[None, :, :], dtype=torch.float32),
    }


def run_one(model: InformerForPrediction, infer_inputs: dict[str, torch.Tensor]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        try:
            out = model.generate(
                past_values=infer_inputs["past_values"],
                past_time_features=infer_inputs["past_time_features"],
                past_observed_mask=infer_inputs["past_observed_mask"],
                future_time_features=infer_inputs["future_time_features"],
            )
        except RuntimeError:
            out = model.generate(
                past_values=infer_inputs["past_values"].unsqueeze(-1),
                past_time_features=infer_inputs["past_time_features"],
                past_observed_mask=infer_inputs["past_observed_mask"].unsqueeze(-1),
                future_time_features=infer_inputs["future_time_features"],
            )
    pred = out.sequences.mean(dim=1).squeeze(0)
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    return pred.cpu().numpy()


def main() -> None:
    st.title("ETF Predictive Allocation")

    model_root = detect_model_root()
    use_model = model_root is not None

    if use_model:
        scaler = joblib.load(model_root / "scaler.joblib")
        meta = json.loads((model_root / "training_meta.json").read_text(encoding="utf-8"))
        model_subdirs = discover_model_subdirs(model_root)
        models = {d: InformerForPrediction.from_pretrained(str(model_root / d)) for d in model_subdirs}
        feature_cols = [c for c in meta.get("feature_cols", BASE_FEATURES)]
        window_size = int(meta.get("window_size", 60))
        pred_len = int(meta.get("pred_len", 5))
    else:
        st.warning("未检测到本地模型文件，当前使用 baseline 预测模式。")
        scaler = None
        models = {}
        feature_cols = BASE_FEATURES
        window_size = 60
        pred_len = 5

    df = load_df(model_root if model_root is not None else Path("."))
    cols = [c for c in feature_cols if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cols] = df[cols].ffill().bfill()

    horizon = st.sidebar.slider("预测步长(天)", min_value=1, max_value=30, value=min(5, pred_len), step=1)

    if use_model and models:
        model_options = list(models.keys())
        if len(model_options) >= 2:
            model_options.append("ensemble")
        model_version = st.sidebar.selectbox("模型版本", model_options)
        x = scaler.transform(df[cols].values)
        infer_inputs = build_inference_tensors(x, window_size=window_size, pred_len=pred_len)
        model_preds: dict[str, np.ndarray] = {}
        for name, m in models.items():
            try:
                model_preds[name] = run_one(m, infer_inputs)
            except Exception:
                continue

        if model_preds:
            if model_version == "ensemble":
                pred = np.mean(np.stack(list(model_preds.values()), axis=0), axis=0)
            elif model_version in model_preds:
                pred = model_preds[model_version]
            else:
                first_ok = next(iter(model_preds.keys()))
                st.warning(f"所选模型 {model_version} {first_ok}。")
                model_version = first_ok
                pred = model_preds[first_ok]
            pred = extend_pred_to_horizon(pred, horizon)
        else:
            st.warning("。")
            model_version = "baseline"
            pred = baseline_predict_returns(df, horizon)
    else:
        model_version = "baseline"
        pred = baseline_predict_returns(df, horizon)

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    last_close = float(close.iloc[-1])
    future_dates = pd.bdate_range(pd.to_datetime(df["Date"].iloc[-1]), periods=horizon + 1)[1:]

    pred_price = [last_close]
    for r in pred:
        pred_price.append(pred_price[-1] * (1 + float(r)))
    pred_price = np.array(pred_price[1:])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.to_datetime(df["Date"]).tail(250), y=close.tail(250), name="Actual"))
    fig.add_trace(go.Scatter(x=future_dates, y=pred_price, name="Predicted"))
    fig.update_layout(template="plotly_dark", title="Price Forecast")
    st.plotly_chart(fig, width="stretch")

    st.caption(f"Model: {model_version} | Root: {str(model_root) if model_root else 'none'}")


if __name__ == "__main__":
    main()
