import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import json
from src.data.preprocess_data import preprocess_data
from src.models.save_load import load_model
import os



def plot_stock_data(ticker, lookback=100, future_days=30):
    
    scaler_path = os.path.join("artifacts", "scalers", ticker, "scaler.npy")

    with open("artifacts/tickers.json", "r") as f:
        nifty50 = json.load(f)

    if ticker not in nifty50:
        raise ValueError(f"{ticker} is not a valid NIFTY 50 stock ticker.")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=20 * 365)
    
    stock_data = yf.download(f"{ticker}.NS", start=start_date, end=end_date)

    df = preprocess_data(stock_data)
    lookback_data = df[-lookback:].values

    scaler = np.load(scaler_path, allow_pickle=True).item()
    scaled_data = scaler.transform(np.array(lookback_data).reshape(-1, 1))

    model = load_model("artifacts/saved_model", ticker)

    temp_input = []
    for i in scaled_data.tolist():
        for j in i:
            temp_input.append(j)

    lst_output = []

    for _ in range(future_days):
        x_input = np.array(temp_input[-100:]).reshape(1, 100, 1)
        yhat = model.predict(x_input, verbose=0)
        lst_output.append(yhat[0][0])
        temp_input.append(yhat[0][0])

    forecasted_prices = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

    past_days = np.arange(1, lookback + 1)
    future_days_idx = np.arange(lookback + 1, lookback + 1 + len(forecasted_prices))

    # === Generate Figure for Streamlit ===
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(past_days, lookback_data, label="Past Data (Last 100)", linewidth=2)
    ax.plot(
        future_days_idx,
        forecasted_prices,
        label="Predicted Future",
        linestyle="--",
        linewidth=2,
    )
    ax.plot(
        [lookback, lookback + 1],
        [lookback_data[-1], forecasted_prices[0]],
        linestyle="dotted",
        linewidth=2,
        color="black",
    )

    ax.set_xlabel("Days")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"Stock Prediction for {ticker}")
    ax.legend()
    ax.grid(True)

    return fig


if __name__ == "__main__":
    plot_stock_data("RELIANCE", lookback=100, future_days=30)
