"""
funding_rate_detector.py
------------------------
Fetches the current BTC perpetual futures funding rate from Binance
and classifies market sentiment based on who is paying whom.

Funding rate > 0: longs pay shorts — market over-leveraged long → bullish sentiment,
                  but at extremes signals liquidation cascade risk (bearish).
Funding rate < 0: shorts pay longs — bearish sentiment, at extremes signals squeeze risk.

Returns
-------
result : dict
    funding_rate        – float  raw rate (e.g. 0.0001 = 0.01 %)
    funding_rate_pct    – float  rate as percentage
    funding_sentiment   – str    "extreme_long" | "bullish" | "neutral" | "bearish" | "extreme_short"
    direction_bias      – str    "bullish" | "bearish" | "neutral"
    funding_score       – int    0–100, how extreme the rate is
    hist_mean_pct       – float  8-period historical mean rate (%)
    next_funding_time   – str    UTC time of next funding settlement
    reason              – str    human-readable explanation

df : pd.DataFrame
    Last 8 funding rate events.
"""

import requests
import pandas as pd
from datetime import datetime, timezone


FAPI_URL = "https://fapi.binance.com"
HIST_LIMIT = 8          # funding events to fetch for context (every 8 h → last ~2.5 days)
EXTREME_THRESHOLD = 0.1  # % — above this = over-leveraged


def _fetch_premium_index(symbol: str) -> dict:
    resp = requests.get(
        f"{FAPI_URL}/fapi/v1/premiumIndex",
        params={"symbol": symbol},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_funding_history(symbol: str, limit: int) -> list:
    resp = requests.get(
        f"{FAPI_URL}/fapi/v1/fundingRate",
        params={"symbol": symbol, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def run_funding_rate_detector(
    symbol: str = "BTCUSDT",
) -> tuple[dict, pd.DataFrame]:

    try:
        premium = _fetch_premium_index(symbol)
        history = _fetch_funding_history(symbol, HIST_LIMIT)
    except Exception as exc:
        return {
            "funding_rate":      0.0,
            "funding_rate_pct":  0.0,
            "funding_sentiment": "neutral",
            "direction_bias":    "neutral",
            "funding_score":     0,
            "hist_mean_pct":     0.0,
            "next_funding_time": "N/A",
            "reason": f"Binance API error: {exc}",
        }, pd.DataFrame()

    current_rate = float(premium.get("lastFundingRate", 0))
    rate_pct     = current_rate * 100

    next_ts   = int(premium.get("nextFundingTime", 0))
    next_time = (
        datetime.fromtimestamp(next_ts / 1000, tz=timezone.utc).strftime("%H:%M UTC")
        if next_ts else "N/A"
    )

    # Historical context
    df = pd.DataFrame(history)
    if not df.empty:
        df["fundingRate"] = df["fundingRate"].astype(float)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        hist_mean_pct = float(df["fundingRate"].mean()) * 100
    else:
        hist_mean_pct = 0.0

    # Sentiment classification
    if rate_pct > EXTREME_THRESHOLD:
        funding_sentiment = "extreme_long"
        direction_bias    = "bearish"   # over-leveraged longs → liquidation cascade risk
    elif rate_pct > 0.01:
        funding_sentiment = "bullish"
        direction_bias    = "bullish"
    elif rate_pct < -EXTREME_THRESHOLD:
        funding_sentiment = "extreme_short"
        direction_bias    = "bullish"   # over-leveraged shorts → squeeze risk
    elif rate_pct < -0.01:
        funding_sentiment = "bearish"
        direction_bias    = "bearish"
    else:
        funding_sentiment = "neutral"
        direction_bias    = "neutral"

    # Score 0–100: how extreme is the rate
    funding_score = min(int(abs(rate_pct) * 500), 100)

    reason = (
        f"Funding rate: {rate_pct:+.4f}% ({funding_sentiment}). "
        f"Positive = longs pay shorts (bullish pressure), negative = shorts pay longs (bearish pressure). "
        f"8-period historical mean: {hist_mean_pct:+.4f}%. "
        f"Next settlement: {next_time}."
    )

    result = {
        "funding_rate":      round(current_rate, 6),
        "funding_rate_pct":  round(rate_pct, 4),
        "funding_sentiment": funding_sentiment,
        "direction_bias":    direction_bias,
        "funding_score":     funding_score,
        "hist_mean_pct":     round(hist_mean_pct, 4),
        "next_funding_time": next_time,
        "reason":            reason,
    }

    return result, df
