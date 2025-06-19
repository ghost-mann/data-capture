import yfinance as yf
from prophet import Prophet
import pandas as pd

# Download 5 years of XAUUSD prices
xau = yf.download("GLD", start="2019-01-01", end="2025-06-18")
xau.to_csv("gld_gold_proxy.csv")