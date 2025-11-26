from src.models.save_load import load_model
from src.models.lstm_model import build_lstm_model, tuned_lstm_model
from src.utils.data_loader import data_loader
from src.models.save_load import save_model
from src.data.fetch_data import fetch_stock_data, save_stock_data
from src.data.preprocess_data import (
    preprocess_data,
    save_preprocessed_data,
    create_dataset,
)
import numpy as np
import keras_tuner as kt
import json
import os


def train(model_name):
    save_stock_data()
    save_preprocessed_data
    x_train, y_train, x_test, y_test = data_loader(model_name)
    model = tuned_lstm_model(x_train, y_train, x_test, y_test)
    save_model(model, "artifacts/saved_model", model_name)
    return model


def predict(input_data, model_name):
    model = load_model("artifacts/saved_model", model_name)
    results = model.predict(input_data)
    return results


if __name__ == "__main__":
    with open("artifacts/tickers.json", "r") as f:
        nifty50 = json.load(f)
    for model_name in nifty50:
        train(model_name)
