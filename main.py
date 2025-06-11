import pandas as pd
import numpy as np
from models.cnn_lstm_autoencoder import build_cnn_lstm_autoencoder
from utils.preprocessing import preprocess
from utils.metrics import evaluate
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv("data/sample_sensor_data.csv")
X_train, X_test, y_test = preprocess(df)

model = build_cnn_lstm_autoencoder(X_train.shape[1:])
model.fit(X_train, X_train, epochs=cfg['train']['epochs'])

reconstructions = model.predict(X_test)
anomaly_scores = np.mean(np.square(X_test - reconstructions), axis=(1, 2))
y_pred = (anomaly_scores > cfg['threshold']['anomaly_score']).astype(int)

print(evaluate(y_test, y_pred))
