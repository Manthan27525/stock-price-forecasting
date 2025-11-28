import streamlit as st
from src.main import train, predict
from src.utils.plot_utils import plot_stock_data
from tensorflow.keras.models import load_model
from src.data.preprocess_data import (
    preprocess_data,
    save_preprocessed_data,
    create_dataset,
)
from src.data.fetch_data import fetch_stock_data, save_stock_data
from src.utils.data_loader import data_loader
from src.models.lstm_model import build_lstm_model, tuned_lstm_model
from src.models.save_load import save_model, load_model
import numpy as np
import json

st.title("Stock Price Forecasting App")
st.write("This app forecasts stock prices using LSTM models.")

with open("artifacts/tickers.json", "r") as f:
    nifty50 = json.load(f)

future_days = st.slider("Select number of days to forecast", 1, 60, 30)
ticker = st.selectbox("Select a NIFTY 50 Stock", nifty50)

if st.button("Tune Selected Model"):
    model = train(ticker)
    st.success(f"Model for {ticker} trained successfully!")
elif st.button("Tune All Models"):
    for model_name in nifty50:
        train(model_name)
    st.success("All models updated successfully!")
elif st.button("Forecast Stock Prices"):
    fig=plot_stock_data(ticker, lookback=100, future_days=future_days)
    st.pyplot(fig)
