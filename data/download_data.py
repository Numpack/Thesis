import requests
import pandas as pd

url = "https://api.binance.com/api/v3/klines"

params = {
    "symbol": "BTCUSDT",
    "interval": "1m",
    "limit": 1000
}


response = requests.get(url, params=params)
data = response.json()

columns = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote",
    "ignore"
]

df = pd.DataFrame(data, columns=columns)
print("Latest candle time:", df.iloc[-1]["open_time"])

print(df.head())
print("Rows:", len(df))

df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

df.to_csv("data/btc_data.csv", index=False)

print("Data saved to btc_data.csv")