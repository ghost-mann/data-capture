import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Download 5 years of XAUUSD prices
xau = yf.download("GLD", start="2019-01-01", end="2025-06-18")
xau.to_csv("gld_gold_proxy.csv")

# Read CSV and reset index to convert Date index to a column
df = pd.read_csv("gld_gold_proxy.csv", skiprows=2)

# rename columns
df.columns = ['ds', 'y', 'High', 'Low', 'Open', 'Volume']

# Create and fit Prophet model
model = Prophet()
model.fit(df)

# Generate future dates for 1 year
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
plt.title("XAUUSD Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price($)")
plt.show()

# Plot forecast components
fig2 = model.plot_components(forecast)
plt.show()

# Print forecast values for next 10 days
print("\nForecast for next 10 days:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))