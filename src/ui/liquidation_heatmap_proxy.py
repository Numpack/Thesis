import requests
import pandas as pd
import numpy as np


FAPI_URL = "https://fapi.binance.com"


def _get_json(endpoint, params=None, timeout=10):
    response = requests.get(
        FAPI_URL + endpoint,
        params=params,
        timeout=timeout
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Binance API error {response.status_code}: {response.text}"
        )

    return response.json()


def fetch_futures_klines(symbol="BTCUSDT", interval="5m", limit=500):
    data = _get_json(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
    )

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    numeric_cols = [
        "open", "high", "low", "close",
        "volume", "quote_volume",
        "taker_buy_base", "taker_buy_quote"
    ]

    df[numeric_cols] = df[numeric_cols].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df


def fetch_open_interest_hist(symbol="BTCUSDT", period="5m", limit=500):
    data = _get_json(
        "/futures/data/openInterestHist",
        {
            "symbol": symbol,
            "period": period,
            "limit": limit
        }
    )

    df = pd.DataFrame(data)

    if df.empty:
        return df

    df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


def merge_price_and_oi(price_df, oi_df):
    if price_df.empty or oi_df.empty:
        return pd.DataFrame()

    price_df = price_df.copy().sort_values("close_time")
    oi_df = oi_df.copy().sort_values("timestamp")

    merged = pd.merge_asof(
        price_df,
        oi_df,
        left_on="close_time",
        right_on="timestamp",
        direction="nearest"
    )

    merged["oi"] = merged["sumOpenInterest"]
    merged["oi_value"] = merged["sumOpenInterestValue"]

    merged["oi_delta"] = merged["oi"].diff()
    merged["oi_value_delta"] = merged["oi_value"].diff()

    merged["candle_direction"] = np.where(
        merged["close"] > merged["open"],
        "up",
        np.where(merged["close"] < merged["open"], "down", "neutral")
    )

    merged = merged.dropna().reset_index(drop=True)

    return merged


def get_liq_price(price, leverage, direction):
    """
    direction = 1  -> long liquidation below price
    direction = -1 -> short liquidation above price
    """
    if leverage <= 0:
        return None

    return price * (1 - direction * (1 / leverage))


def get_price_bin(price, price_bin_size):
    return round(price / price_bin_size) * price_bin_size


def add_or_merge_level(levels, new_level, storage_size=500):
    for lvl in levels:
        if (
            lvl["price_bin"] == new_level["price_bin"]
            and lvl["side"] == new_level["side"]
        ):
            total_contracts = lvl["contracts"] + new_level["contracts"]

            if total_contracts > 0:
                lvl["price"] = (
                    lvl["price"] * lvl["contracts"]
                    + new_level["price"] * new_level["contracts"]
                ) / total_contracts

            lvl["contracts"] = total_contracts
            lvl["last_time"] = new_level["last_time"]
            lvl["entries"] += 1
            return levels

    levels.append(new_level)

    if len(levels) > storage_size:
        levels.pop(0)

    return levels


def remove_squeezed_levels(levels, high, low, event_time):
    active = []
    squeezed = []

    for lvl in levels:
        if lvl["side"] == "long_liquidation" and low <= lvl["price"]:
            lvl = lvl.copy()
            lvl["squeezed"] = True
            lvl["squeezed_time"] = event_time
            squeezed.append(lvl)

        elif lvl["side"] == "short_liquidation" and high >= lvl["price"]:
            lvl = lvl.copy()
            lvl["squeezed"] = True
            lvl["squeezed_time"] = event_time
            squeezed.append(lvl)

        else:
            active.append(lvl)

    return active, squeezed


def build_proxy_levels(
    merged_df,
    leverages=None,
    dispersion_factor=0.20,
    price_bin_size=50,
    storage_size=500,
    filter_peaks=False,
    signal_length=200,
    signal_mult=1.0
):
    if leverages is None:
        leverages = [125, 100, 50]

    leverages = [lev for lev in leverages if lev > 0]

    if not leverages or merged_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = merged_df.copy()

    df["oi_abs"] = df["oi_delta"].abs()
    df["oi_signal"] = (
        df["oi_abs"].rolling(signal_length, min_periods=20).mean()
        + signal_mult * df["oi_abs"].rolling(signal_length, min_periods=20).std()
    )

    total_leverage = sum(leverages)

    active_levels = []
    squeezed_levels = []

    for _, row in df.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        open_price = float(row["open"])
        close_price = float(row["close"])
        oi_delta = float(row["oi_delta"])
        event_time = row["close_time"]

        active_levels, squeezed_now = remove_squeezed_levels(
            active_levels,
            high=high,
            low=low,
            event_time=event_time
        )

        squeezed_levels.extend(squeezed_now)

        if oi_delta <= 0:
            continue

        if filter_peaks:
            if pd.isna(row["oi_signal"]) or abs(oi_delta) <= row["oi_signal"]:
                continue

        if close_price == open_price:
            continue

        low_src = (low + open_price) / 2
        high_src = (high + open_price) / 2

        for lev in leverages:
            lev_weight = lev / total_leverage

            if close_price > open_price:
                main_side = "long_liquidation"
                main_price = get_liq_price(low_src, lev, direction=1)
                main_contracts = oi_delta * (1 - dispersion_factor) * lev_weight

                active_levels = add_or_merge_level(
                    active_levels,
                    {
                        "side": main_side,
                        "price": main_price,
                        "price_bin": get_price_bin(main_price, price_bin_size),
                        "contracts": main_contracts,
                        "leverage": lev,
                        "first_time": event_time,
                        "last_time": event_time,
                        "entries": 1,
                        "squeezed": False
                    },
                    storage_size=storage_size
                )

                if dispersion_factor > 0:
                    dispersion_side = "short_liquidation"
                    dispersion_price = get_liq_price(high_src, lev, direction=-1)
                    dispersion_contracts = oi_delta * dispersion_factor * lev_weight

                    active_levels = add_or_merge_level(
                        active_levels,
                        {
                            "side": dispersion_side,
                            "price": dispersion_price,
                            "price_bin": get_price_bin(dispersion_price, price_bin_size),
                            "contracts": dispersion_contracts,
                            "leverage": lev,
                            "first_time": event_time,
                            "last_time": event_time,
                            "entries": 1,
                            "squeezed": False
                        },
                        storage_size=storage_size
                    )

            elif close_price < open_price:
                main_side = "short_liquidation"
                main_price = get_liq_price(high_src, lev, direction=-1)
                main_contracts = oi_delta * (1 - dispersion_factor) * lev_weight

                active_levels = add_or_merge_level(
                    active_levels,
                    {
                        "side": main_side,
                        "price": main_price,
                        "price_bin": get_price_bin(main_price, price_bin_size),
                        "contracts": main_contracts,
                        "leverage": lev,
                        "first_time": event_time,
                        "last_time": event_time,
                        "entries": 1,
                        "squeezed": False
                    },
                    storage_size=storage_size
                )

                if dispersion_factor > 0:
                    dispersion_side = "long_liquidation"
                    dispersion_price = get_liq_price(low_src, lev, direction=1)
                    dispersion_contracts = oi_delta * dispersion_factor * lev_weight

                    active_levels = add_or_merge_level(
                        active_levels,
                        {
                            "side": dispersion_side,
                            "price": dispersion_price,
                            "price_bin": get_price_bin(dispersion_price, price_bin_size),
                            "contracts": dispersion_contracts,
                            "leverage": lev,
                            "first_time": event_time,
                            "last_time": event_time,
                            "entries": 1,
                            "squeezed": False
                        },
                        storage_size=storage_size
                    )

    active_df = pd.DataFrame(active_levels)
    squeezed_df = pd.DataFrame(squeezed_levels)

    return active_df, squeezed_df


def analyze_active_levels(active_df, current_price):
    if active_df.empty:
        return {
            "liquidation_imbalance_score": 0.0,
            "liquidity_attraction": "neutral",
            "dominant_liquidity_side": "neutral",
            "main_zone_price": None,
            "main_zone_side": "neutral",
            "main_zone_strength": 0.0,
            "upper_liquidation_contracts": 0.0,
            "lower_liquidation_contracts": 0.0,
            "upper_share": 0.0,
            "lower_share": 0.0,
            "reason": "No active estimated liquidation levels."
        }, pd.DataFrame()

    zones = (
        active_df
        .groupby(["price_bin", "side"], observed=True)
        .agg(
            zone_price=("price", "mean"),
            contracts=("contracts", "sum"),
            entries=("entries", "sum"),
            avg_leverage=("leverage", "mean"),
            max_leverage=("leverage", "max"),
            first_time=("first_time", "min"),
            last_time=("last_time", "max")
        )
        .reset_index()
    )

    zones["distance_abs"] = abs(zones["zone_price"] - current_price)
    zones["distance_pct"] = zones["distance_abs"] / current_price * 100

    max_contracts = zones["contracts"].max()

    if max_contracts > 0:
        zones["zone_strength"] = zones["contracts"] / max_contracts * 100
    else:
        zones["zone_strength"] = 0.0

    zones["impact_score"] = zones["zone_strength"] / (
        (zones["distance_pct"] + 0.15) ** 1.15
    )

    upper = zones[zones["zone_price"] > current_price]
    lower = zones[zones["zone_price"] < current_price]

    upper_contracts = upper["contracts"].sum()
    lower_contracts = lower["contracts"].sum()

    total = upper_contracts + lower_contracts

    if total > 0:
        upper_share = upper_contracts / total * 100
        lower_share = lower_contracts / total * 100
    else:
        upper_share = 0.0
        lower_share = 0.0

    imbalance = upper_share - lower_share

    if imbalance > 10:
        liquidity_attraction = "up"
        dominant_liquidity_side = "above"
    elif imbalance < -10:
        liquidity_attraction = "down"
        dominant_liquidity_side = "below"
    else:
        liquidity_attraction = "neutral"
        dominant_liquidity_side = "balanced"

    zones = zones.sort_values("impact_score", ascending=False).reset_index(drop=True)
    main_zone = zones.iloc[0]

    if main_zone["zone_price"] > current_price:
        main_side = "above"
    elif main_zone["zone_price"] < current_price:
        main_side = "below"
    else:
        main_side = "at_price"

    result = {
        "liquidation_imbalance_score": round(float(abs(imbalance)), 2),
        "liquidity_attraction": liquidity_attraction,
        "dominant_liquidity_side": dominant_liquidity_side,

        "main_zone_price": round(float(main_zone["zone_price"]), 2),
        "main_zone_side": main_side,
        "main_zone_strength": round(float(main_zone["zone_strength"]), 2),

        "upper_liquidation_contracts": round(float(upper_contracts), 4),
        "lower_liquidation_contracts": round(float(lower_contracts), 4),
        "upper_share": round(float(upper_share), 2),
        "lower_share": round(float(lower_share), 2),

        "reason": (
            "Estimated liquidation zones are calculated from positive open-interest delta, "
            "candle direction and assumed leverage distribution. "
            f"Upper estimated contracts: {upper_contracts:.4f}. "
            f"Lower estimated contracts: {lower_contracts:.4f}. "
            f"Possible liquidity attraction: {liquidity_attraction.upper()}."
        )
    }

    return result, zones


def run_liquidation_heatmap_proxy(
    symbol="BTCUSDT",
    interval="5m",
    oi_period="5m",
    limit=500,
    leverages=None,
    dispersion_factor=0.20,
    price_bin_size=50,
    storage_size=500,
    filter_peaks=False,
    signal_length=200,
    signal_mult=1.0
):
    if leverages is None:
        leverages = [125, 100, 50]

    price_df = fetch_futures_klines(
        symbol=symbol,
        interval=interval,
        limit=limit
    )

    oi_df = fetch_open_interest_hist(
        symbol=symbol,
        period=oi_period,
        limit=limit
    )

    merged = merge_price_and_oi(price_df, oi_df)

    if merged.empty:
        empty_result = {
            "liquidation_imbalance_score": 0.0,
            "liquidity_attraction": "neutral",
            "dominant_liquidity_side": "neutral",
            "main_zone_price": None,
            "main_zone_side": "neutral",
            "main_zone_strength": 0.0,
            "upper_liquidation_contracts": 0.0,
            "lower_liquidation_contracts": 0.0,
            "upper_share": 0.0,
            "lower_share": 0.0,
            "current_price": None,
            "reason": "Could not merge futures price data with open interest history."
        }

        return empty_result, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    active_levels, squeezed_levels = build_proxy_levels(
        merged_df=merged,
        leverages=leverages,
        dispersion_factor=dispersion_factor,
        price_bin_size=price_bin_size,
        storage_size=storage_size,
        filter_peaks=filter_peaks,
        signal_length=signal_length,
        signal_mult=signal_mult
    )

    current_price = float(price_df["close"].iloc[-1])

    result, zones = analyze_active_levels(
        active_levels,
        current_price=current_price
    )

    result["current_price"] = round(float(current_price), 2)
    result["symbol"] = symbol
    result["interval"] = interval
    result["oi_period"] = oi_period
    result["leverages"] = leverages
    result["dispersion_factor"] = dispersion_factor
    result["price_bin_size"] = price_bin_size

    return result, merged, active_levels, zones, squeezed_levels