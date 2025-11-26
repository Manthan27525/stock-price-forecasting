import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta


def fetch_stock_data(
    name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    name = yf.download(name, start=start_date, end=end_date)
    return name


nifty50 = [
    "ADANIENT",  # Adani Enterprises Ltd :contentReference[oaicite:1]{index=1}
    "ADANIPORTS",  # Adani Ports & SEZ Ltd :contentReference[oaicite:2]{index=2}
    "APOLLOHOSP",  # Apollo Hospitals Enterprise Ltd :contentReference[oaicite:3]{index=3}
    "ASIANPAINT",  # Asian Paints Ltd :contentReference[oaicite:4]{index=4}
    "AXISBANK",  # Axis Bank Ltd :contentReference[oaicite:5]{index=5}
    "BAJAJ-AUTO",  # Bajaj Auto Ltd :contentReference[oaicite:6]{index=6}
    "BAJFINANCE",  # Bajaj Finance Ltd :contentReference[oaicite:7]{index=7}
    "BAJAJFINSV",  # Bajaj Finserv Ltd :contentReference[oaicite:8]{index=8}
    "BEL",  # Bharat Electronics Ltd :contentReference[oaicite:9]{index=9}
    "BHARTIARTL",  # Bharti Airtel Ltd :contentReference[oaicite:10]{index=10}
    "COALINDIA",  # Coal India Ltd :contentReference[oaicite:11]{index=11}
    "DIVISLAB",  # Divi’s Laboratories Ltd :contentReference[oaicite:12]{index=12}
    "DRREDDY",  # Dr. Reddy’s Laboratories Ltd :contentReference[oaicite:13]{index=13}
    "EICHERMOT",  # Eicher Motors Ltd :contentReference[oaicite:14]{index=14}
    "GRASIM",  # Grasim Industries Ltd :contentReference[oaicite:15]{index=15}
    "HCLTECH",  # HCL Technologies Ltd :contentReference[oaicite:16]{index=16}
    "HDFCBANK",  # HDFC Bank Ltd :contentReference[oaicite:17]{index=17}
    "HINDALCO",  # Hindalco Industries Ltd :contentReference[oaicite:18]{index=18}
    "HINDUNILVR",  # Hindustan Unilever Ltd :contentReference[oaicite:19]{index=19}
    "ICICIBANK",  # ICICI Bank Ltd :contentReference[oaicite:20]{index=20}
    "INDUSINDBK",  # IndusInd Bank Ltd :contentReference[oaicite:21]{index=21}
    "INFY",  # Infosys Ltd :contentReference[oaicite:22]{index=22}
    "IOC",  # Indian Oil Corporation Ltd (if present) – please verify; alternatively other company :contentReference[oaicite:23]{index=23}
    "ITC",  # ITC Ltd :contentReference[oaicite:24]{index=24}
    "JSWSTEEL",  # JSW Steel Ltd :contentReference[oaicite:25]{index=25}
    "KOTAKBANK",  # Kotak Mahindra Bank Ltd :contentReference[oaicite:26]{index=26}
    "LT",  # Larsen & Toubro Ltd :contentReference[oaicite:27]{index=27}
    "LTIM",  # L&T Technology + Mindtree merged entity (if applicable) :contentReference[oaicite:28]{index=28}
    "MARUTI",  # Maruti Suzuki India Ltd :contentReference[oaicite:29]{index=29}
    "M&M",  # Mahindra & Mahindra Ltd :contentReference[oaicite:30]{index=30}
    "NESTLEIND",  # Nestle India Ltd :contentReference[oaicite:31]{index=31}
    "NTPC",  # NTPC Ltd :contentReference[oaicite:32]{index=32}
    "ONGC",  # Oil & Natural Gas Corporation Ltd :contentReference[oaicite:33]{index=33}
    "POWERGRID",  # Power Grid Corporation of India Ltd :contentReference[oaicite:34]{index=34}
    "RELIANCE",  # Reliance Industries Ltd :contentReference[oaicite:35]{index=35}
    "SBIN",  # State Bank of India Ltd :contentReference[oaicite:36]{index=36}
    "SHREECEM",  # Shree Cement Ltd (verify inclusion) :contentReference[oaicite:37]{index=37}
    "SUNPHARMA",  # Sun Pharmaceutical Industries Ltd :contentReference[oaicite:38]{index=38}
    "TATACONSUM",  # Tata Consumer Products Ltd :contentReference[oaicite:39]{index=39}
    "TATAMOTORS",  # Tata Motors Ltd :contentReference[oaicite:40]{index=40}
    "TATASTEEL",  # Tata Steel Ltd :contentReference[oaicite:41]{index=41}
    "TCS",  # Tata Consultancy Services Ltd :contentReference[oaicite:42]{index=42}
    "TECHM",  # Tech Mahindra Ltd :contentReference[oaicite:43]{index=43}
    "TITAN",  # Titan Company Ltd :contentReference[oaicite:44]{index=44}
    "ULTRACEMCO",  # UltraTech Cement Ltd :contentReference[oaicite:45]{index=45}
    "UPL",  # UPL Ltd (verify inclusion) :contentReference[oaicite:46]{index=46}
    "WIPRO",  # Wipro Ltd :contentReference[oaicite:47]{index=47}
    "ZEEL",  # Zee Entertainment Enterprises Ltd (or substitute if changed) :contentReference[oaicite:48]{index=48}
]

with open("artifacts/tickers.json", "w") as f:
    json.dump(nifty50, f, indent=4)


def save_stock_data():
    with open("artifacts/tickers.json", "r") as f:
        nifty50 = json.load(f)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20 * 365)
    for i in nifty50:
        data = fetch_stock_data(f"{i}.NS", start_date, end_date)
        data.to_csv(f"artifacts/NIFTY50/{i}.csv")


if __name__ == "__main__":
    save_stock_data()
