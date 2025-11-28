import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import json


def preprocess_data(df):
    data = pd.DataFrame(columns=["Date", "Close", "High", "Low", "Open", "Volume"])
    data = df.iloc[2:, :]
    data = data.reset_index()["Close"]
    return data.reset_index(drop=True)


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i : (i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def save_preprocessed_data(ticker):
    file_path = f"artifacts/NIFTY50/{ticker}.csv"
    df = pd.read_csv(file_path)
    df = preprocess_data(df)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(np.array(df).reshape(-1, 1))

    train_data, test_data = train_test_split(
        data, test_size=0.35, random_state=42, shuffle=False
    )
    x_train, y_train = create_dataset(train_data, time_step=100)
    x_test, y_test = create_dataset(test_data, time_step=100)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    os.makedirs(f"artifacts/processed data/{ticker}", exist_ok=True)
    np.save(f"artifacts/processed data/{ticker}/x_train.npy", x_train)
    np.save(f"artifacts/processed data/{ticker}/y_train.npy", y_train)
    np.save(f"artifacts/processed data/{ticker}/x_test.npy", x_test)
    np.save(f"artifacts/processed data/{ticker}/y_test.npy", y_test)

    os.makedirs(f"artifacts/scalers/{ticker}", exist_ok=True)
    np.save(f"artifacts/scalers/{ticker}/scaler.npy", scaler)


if __name__ == "__main__":
    with open("artifacts/tickers.json", "r") as f:
        nifty50 = json.load(f)
        for ticker in nifty50:
            save_preprocessed_data(ticker)
