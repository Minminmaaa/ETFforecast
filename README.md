# ETF Deep Learning Allocation Project

运行 notebooks/train.ipynb 即可完成训练与模型落地。
如果 Hugging Face 数据集不可用，会自动降级到 yfinance 数据源。
模型会保存到 model/informer、model/scaler.joblib、model/training_meta.json。
如需 2-3 个模型，可额外放置 model/informer_v2、model/informer_v3，应用会自动识别并支持集成预测。
