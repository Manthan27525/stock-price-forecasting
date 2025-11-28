from src.models.save_load import load_model
from src.models.lstm_model import simple_lstm_model
from src.utils.data_loader import data_loader
from src.models.save_load import save_model
from src.data.fetch_data import fetch_stock_data, save_stock_data
from src.data.preprocess_data import (
    preprocess_data,
    save_preprocessed_data,
    create_dataset,
)
import numpy as np

import json
import os

def simple_train(model_name):
    os.makedirs("artifacts/saved_model", exist_ok=True)
    save_stock_data(model_name)
    save_preprocessed_data(model_name)
    x_train, y_train, x_test, y_test = data_loader(model_name)
    model = simple_lstm_model(x_train, y_train, x_test, y_test)
    model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
    save_model(model, "artifacts/saved_model", model_name)
    print(f"Model for {model_name} trained and saved successfully.")


if __name__ == "__main__":
    with open("artifacts/tickers.json", "r") as f:
        nifty50 = json.load(f)
    for model_name in nifty50:
        simple_train(model_name)
