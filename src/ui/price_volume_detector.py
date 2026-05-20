"""
price_volume_detector.py
------------------------
Detects suspicious price/volume behaviour on BTCUSDT futures using
an Isolation Forest trained on recent candle data.

Returns
-------
result : dict
    anomaly_score          – float, mean anomaly strength of recent window
    recent_anomaly_count   – int,   number of anomalies in the last 20 candles
    volatility_regime      – str,   "NORMAL" | "ELEVATED" | "ANOMALOUS"
    direction              – str,   "bullish" | "bearish" | "neutral"
    latest_price_return    – float
    latest_volatility      – float
    latest_raw_score       – float
    reason                 – str,   human-readable explanation

df : pd.DataFrame
    Enriched candle dataframe (features + model outputs).
"""

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ── helpers ──────────────────────────────────────────────────────────────────

def _fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Pull klines from Binance Futures public REST API."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)

    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used by the Isolation Forest."""
    df = df.copy()

    # ── price features ───────────────────────────────────────────────────────
    df["price_return"]  = df["close"].pct_change()
    df["candle_range"]  = (df["high"] - df["low"]) / df["close"]
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)

    # ── volume features ──────────────────────────────────────────────────────
    df["volume_change"] = df["volume"].pct_change()

    roll = df["volume"].rolling(20, min_periods=5)
    df["volume_z_score"] = (df["volume"] - roll.mean()) / (roll.std() + 1e-9)

    df["taker_buy_ratio"] = df["taker_buy_base"] / (df["volume"] + 1e-9)

    # OFI: positive = net buying pressure, negative = net selling pressure
    df["ofi"] = df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])

    # ── volatility ───────────────────────────────────────────────────────────
    df["volatility"] = df["price_return"].rolling(20, min_periods=5).std()

    # ── momentum ─────────────────────────────────────────────────────────────
    df["momentum_3"]  = df["close"].pct_change(3)
    df["momentum_10"] = df["close"].pct_change(10)

    return df.dropna()


FEATURE_COLS = [
    "price_return",
    "candle_range",
    "close_position",
    "volume_change",
    "volume_z_score",
    "taker_buy_ratio",
    "volatility",
    "momentum_3",
    "momentum_10",
]


# ── main entry point ─────────────────────────────────────────────────────────

def run_price_volume_detector(
    symbol: str  = "BTCUSDT",
    interval: str = "1m",
    limit: int    = 500,
    contamination: float = 0.05,
    n_estimators: int    = 200,
    recent_window: int   = 100,
) -> tuple[dict, pd.DataFrame]:
    """
    Parameters
    ----------
    symbol        : Binance futures pair, e.g. "BTCUSDT"
    interval      : kline interval, e.g. "1m", "5m"
    limit         : number of candles to fetch (max 1500)
    contamination : expected fraction of anomalies (Isolation Forest param)
    n_estimators  : number of trees in the forest
    recent_window : candles used for the summary statistics

    Returns
    -------
    (result_dict, enriched_dataframe)
    """

    # 1. fetch ────────────────────────────────────────────────────────────────
    try:
        df_raw = _fetch_klines(symbol, interval, limit)
    except Exception as exc:
        empty = pd.DataFrame()
        return {
            "anomaly_score": 0.0,
            "recent_anomaly_count": 0,
            "volatility_regime": "UNKNOWN",
            "direction": "neutral",
            "latest_price_return": 0.0,
            "latest_volatility": 0.0,
            "latest_raw_score": 0.0,
            "reason": f"Data fetch failed: {exc}",
        }, empty

    # 2. feature engineering ──────────────────────────────────────────────────
    df = _engineer_features(df_raw)

    if len(df) < 50:
        return {
            "anomaly_score": 0.0,
            "recent_anomaly_count": 0,
            "volatility_regime": "UNKNOWN",
            "direction": "neutral",
            "latest_price_return": 0.0,
            "latest_volatility": 0.0,
            "latest_raw_score": 0.0,
            "reason": "Not enough data after feature engineering.",
        }, df

    # 3. scale + fit Isolation Forest ─────────────────────────────────────────
    X = df[FEATURE_COLS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # raw_score: more negative → more anomalous
    raw_scores = model.score_samples(X_scaled)
    labels     = model.predict(X_scaled)          # -1 anomaly, +1 normal

    df["raw_score"]       = raw_scores
    df["anomaly_label"]   = labels
    df["is_anomaly"]      = labels == -1

    # normalise raw_score to [0, 1] where 1 = most anomalous
    s_min, s_max = raw_scores.min(), raw_scores.max()
    df["anomaly_strength"] = (raw_scores - s_max) / (s_min - s_max + 1e-9)

    # 4. summary statistics (recent window) ───────────────────────────────────
    recent = df.tail(recent_window)

    recent_anomaly_count = int(recent["is_anomaly"].sum())
    anomaly_score        = float(recent["anomaly_strength"].mean())
    latest_return        = float(df["price_return"].iloc[-1])
    latest_volatility    = float(df["volatility"].iloc[-1])
    latest_raw_score     = float(df["raw_score"].iloc[-1])

    # 5. volatility regime ────────────────────────────────────────────────────
    vol_series    = df["volatility"].dropna()
    # exclude recent window so current volatility is evaluated against historical baseline
    vol_baseline  = vol_series.iloc[:-recent_window] if len(vol_series) > recent_window * 2 else vol_series
    vol_mean      = vol_baseline.mean()
    vol_std       = vol_baseline.std()
    vol_threshold_elevated  = vol_mean + 1.0 * vol_std
    vol_threshold_anomalous = vol_mean + 2.0 * vol_std

    if latest_volatility >= vol_threshold_anomalous:
        volatility_regime = "ANOMALOUS"
    elif latest_volatility >= vol_threshold_elevated:
        volatility_regime = "ELEVATED"
    else:
        volatility_regime = "NORMAL"

    # 6. direction bias — Order Flow Imbalance (OFI)
    # OFI = taker_buy_volume - taker_sell_volume, normalised by total volume → [-1, +1]
    # Positive = buyers more aggressive, negative = sellers more aggressive
    recent_ofi   = recent["ofi"].sum()
    recent_vol   = recent["volume"].sum()
    ofi_ratio    = float(recent_ofi / (recent_vol + 1e-9))

    if ofi_ratio > 0.04:
        direction = "bullish"
    elif ofi_ratio < -0.04:
        direction = "bearish"
    else:
        direction = "neutral"

    # 7. human-readable reason ────────────────────────────────────────────────
    anomaly_pct = (recent_anomaly_count / recent_window) * 100
    reason_parts = [
        f"In the last {recent_window} candles, {recent_anomaly_count} ({anomaly_pct:.0f}%) "
        f"were flagged as anomalous by the Isolation Forest.",
        f"Current volatility ({latest_volatility:.5f}) puts the regime in {volatility_regime}.",
        f"Order Flow Imbalance over the window: {ofi_ratio:+.4f} (>0 = buying pressure), suggesting a {direction} bias.",
    ]

    if volatility_regime == "ANOMALOUS":
        reason_parts.append(
            "⚠️ Volatility is significantly above the historical mean — "
            "unusual price/volume behaviour detected."
        )
    elif volatility_regime == "ELEVATED":
        reason_parts.append(
            "⚡ Volatility is moderately elevated compared to the historical baseline."
        )

    reason = " ".join(reason_parts)

    result = {
        "anomaly_score":        round(anomaly_score, 4),
        "recent_anomaly_count": recent_anomaly_count,
        "volatility_regime":    volatility_regime,
        "direction":            direction,
        "ofi_ratio":            round(ofi_ratio, 4),
        "latest_price_return":  round(latest_return, 6),
        "latest_volatility":    round(latest_volatility, 6),
        "latest_raw_score":     round(latest_raw_score, 4),
        "reason":               reason,
    }

    return result, df