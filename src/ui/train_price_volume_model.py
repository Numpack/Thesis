import requests
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MODEL_PATH = "price_volume_isolation_forest.pkl"

FEATURES = [
    "price_return",
    "candle_range",
    "volume",
    "volume_change",
    "volume_z_score",
    "volatility"
]


def fetch_ohlcv(symbol="BTCUSDT", interval="1m", limit=1000):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df


def add_features(df, window=20):
    df = df.copy()

    df["price_return"] = df["close"].pct_change()
    df["candle_range"] = (df["high"] - df["low"]) / df["open"]
    df["volume_change"] = df["volume"].pct_change()

    df["rolling_volume_mean"] = df["volume"].rolling(window).mean()
    df["rolling_volume_std"] = df["volume"].rolling(window).std()

    df["volume_z_score"] = (
        (df["volume"] - df["rolling_volume_mean"]) /
        df["rolling_volume_std"].replace(0, np.nan)
    )

    df["volatility"] = df["price_return"].rolling(window).std()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


def train_model():
    df = fetch_ohlcv(
        symbol="BTCUSDT",
        interval="1m",
        limit=1000
    )

    df = add_features(df)

    X = df[FEATURES]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("isolation_forest", IsolationForest(
            n_estimators=200,
            contamination=0.03,
            random_state=42
        ))
    ])

    model.fit(X)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Training rows: {len(df)}")


if __name__ == "__main__":
    train_model()