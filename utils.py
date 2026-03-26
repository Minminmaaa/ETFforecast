import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from transformers import InformerConfig, InformerForPrediction
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Tuple

# 复用 notebook 中的定义
BASE_FEATURES = ['Open','High','Low','Close','Volume','CPI','Unemployment Rate','DXY','Gold/Copper Ratio']

@dataclass
class WindowConfig:
    window_size: int = 60
    label_len: int = 30
    pred_len: int = 5

class InformerWindowDataset(Dataset):
    # 与原 notebook 相同
    ...

class InformerDataCollator:
    # 与原 notebook 相同
    ...

def load_etf_dataframe(dataset_name='P2SAMAPA/my-etf-data'):
    # 复制 notebook 中的函数，但去掉 yfinance 和合成数据的 fallback（可选）
    # 如果使用合成数据，需在本地生成，或从 HuggingFace 加载
    ...

def add_target(df, pred_len=5):
    ...

def split_train_val_test(df, train_ratio=0.7, val_ratio=0.15):
    ...

def rmse_np(y_true, y_pred):
    ...