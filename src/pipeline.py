from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


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


def _find_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    lookup = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        key = name.lower().strip()
        if key in lookup:
            return lookup[key]
    return None


def load_etf_dataframe(dataset_name: str = "P2SAMAPA/my-etf-data") -> pd.DataFrame:
    ds = load_dataset(dataset_name)
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    date_col = _find_column(df, ["Date", "date", "timestamp"])
    if date_col is None:
        raise ValueError("No date column found in dataset.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "Date"})

    col_map = {}
    for c in ["Open", "High", "Low", "Close", "Volume", "CPI", "Unemployment Rate", "DXY"]:
        found = _find_column(df, [c, c.replace(" ", "_"), c.lower(), c.lower().replace(" ", "_")])
        if found is not None and found != c:
            col_map[found] = c
    if col_map:
        df = df.rename(columns=col_map)

    if "Gold/Copper Ratio" not in df.columns:
        gold_col = _find_column(df, ["Gold", "Gold Price", "gold_price", "XAUUSD"])
        copper_col = _find_column(df, ["Copper", "Copper Price", "copper_price", "HG"])
        if gold_col is not None and copper_col is not None:
            denom = pd.to_numeric(df[copper_col], errors="coerce").replace(0, np.nan)
            df["Gold/Copper Ratio"] = pd.to_numeric(df[gold_col], errors="coerce") / denom
        else:
            df["Gold/Copper Ratio"] = np.nan

    for c in BASE_FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    numeric_cols = [c for c in BASE_FEATURES if c != "Gold/Copper Ratio"] + ["Gold/Copper Ratio"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def add_target(df: pd.DataFrame, pred_len: int = 5) -> pd.DataFrame:
    out = df.copy()
    out[f"Ret_t+{pred_len}"] = out["Close"].shift(-pred_len) / out["Close"] - 1.0
    return out.dropna(subset=[f"Ret_t+{pred_len}"]).reset_index(drop=True)


def split_train_val_test(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def fit_and_save_scaler(train_df: pd.DataFrame, feature_cols: List[str], model_dir: str = "./model") -> StandardScaler:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    joblib.dump(scaler, model_path / "scaler.joblib")
    return scaler


def transform_features(df: pd.DataFrame, scaler: StandardScaler, feature_cols: List[str]) -> np.ndarray:
    return scaler.transform(df[feature_cols].values)


@dataclass
class WindowConfig:
    window_size: int = 60
    label_len: int = 30
    pred_len: int = 5


class InformerWindowDataset(Dataset):
    def __init__(
        self,
        scaled_features: np.ndarray,
        target_values: np.ndarray,
        cfg: WindowConfig,
    ) -> None:
        self.X = scaled_features.astype(np.float32)
        self.y = target_values.astype(np.float32)
        self.cfg = cfg

        min_size = cfg.window_size + cfg.pred_len
        if len(self.X) < min_size:
            raise ValueError(f"Dataset too small for windows: need >= {min_size}, got {len(self.X)}")

    def __len__(self) -> int:
        return len(self.X) - self.cfg.window_size - self.cfg.pred_len + 1

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        ws = self.cfg.window_size
        pl = self.cfg.pred_len

        left = idx
        right = idx + ws
        future_right = right + pl

        context_features = self.X[left:right]  # [ws, n_features]
        future_features = self.X[right:future_right]  # [pl, n_features]
        past_values = context_features[:, 3:4]  # scaled Close as target source [ws, 1]
        future_values = self.y[right:future_right].reshape(pl, 1)

        return {
            "past_values": past_values,
            "past_time_features": context_features,
            "past_observed_mask": np.ones_like(past_values, dtype=np.float32),
            "future_values": future_values,
            "future_time_features": future_features,
        }


class InformerDataCollator:
    def __call__(self, batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        keys = batch[0].keys()
        result: Dict[str, torch.Tensor] = {}
        for k in keys:
            result[k] = torch.tensor(np.stack([item[k] for item in batch], axis=0), dtype=torch.float32)
        return result


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def save_training_meta(
    feature_cols: List[str],
    cfg: WindowConfig,
    model_dir: str = "./model",
) -> None:
    meta = {
        "feature_cols": feature_cols,
        "window_size": cfg.window_size,
        "label_len": cfg.label_len,
        "pred_len": cfg.pred_len,
    }
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    with open(model_path / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_training_meta(model_dir: str = "./model") -> Dict[str, int | List[str]]:
    with open(Path(model_dir) / "training_meta.json", "r", encoding="utf-8") as f:
        return json.load(f)
