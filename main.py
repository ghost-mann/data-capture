import yfinance as yf
from prophet import Prophet
import pandas as pd

# Download 5 years of XAUUSD prices
xau = yf.download("GLD", start="2019-01-01", end="2025-06-18")
xau.to_csv("gld_gold_proxy.csv")

# store csv values in data frame
df = pd.read_csv("gld_gold_proxy.csv")
df = df.rename(columns={"Date":"ds", "Close":"y"})
df["ds"] = pd.to_datetime(df["ds"])

model = Prophet()
model.fit(df)

