"""
whale_activity_detector.py
--------------------------
Detects large ("whale") Bitcoin transactions using the public
mempool.space REST API — no API key required.

Returns
-------
result : dict
    whale_score              – int   0-100 composite score
    whale_activity           – str   "low" | "medium" | "high" | "extreme"
    whale_count              – int   number of whale transactions detected
    largest_transaction_btc  – float size of the single largest tx (BTC)
    risk_level               – str   "low" | "medium" | "high"
    reason                   – str   human-readable explanation

df : pd.DataFrame
    Table of detected whale transactions (empty if none found).
"""

import requests
import pandas as pd


# ── constants ─────────────────────────────────────────────────────────────────

MEMPOOL_BASE   = "https://mempool.space/api"
SATOSHI        = 1e-8          # satoshi → BTC conversion

# How many recent mempool transactions to inspect
TX_FETCH_LIMIT = 25


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_recent_txids(limit: int = TX_FETCH_LIMIT) -> list[str]:
    """Return a list of recent unconfirmed tx IDs from the mempool."""
    url = f"{MEMPOOL_BASE}/mempool/recent"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # each item has "txid" key
    return [item["txid"] for item in data[:limit]]


def _get_tx_details(txid: str) -> dict | None:
    """Fetch full transaction details for a single txid."""
    url = f"{MEMPOOL_BASE}/tx/{txid}"
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _tx_value_btc(tx: dict) -> float:
    """Sum all output values (satoshi → BTC)."""
    try:
        total_sat = sum(v.get("value", 0) for v in tx.get("vout", []))
        return total_sat * SATOSHI
    except Exception:
        return 0.0


def _fee_rate(tx: dict) -> float:
    """Return fee rate in sat/vbyte, or 0 if unavailable."""
    try:
        fee   = tx.get("fee", 0)
        vsize = tx.get("size", tx.get("weight", 4) // 4)
        return fee / vsize if vsize else 0.0
    except Exception:
        return 0.0


# ── main entry point ──────────────────────────────────────────────────────────

def run_whale_activity_detector(
    threshold_btc: float = 10.0,
) -> tuple[dict, pd.DataFrame]:
    """
    Parameters
    ----------
    threshold_btc : minimum BTC value for a transaction to be considered a whale tx.

    Returns
    -------
    (result_dict, whale_dataframe)
    """

    # 1. fetch recent mempool tx ids ──────────────────────────────────────────
    try:
        txids = _get_recent_txids(TX_FETCH_LIMIT)
    except Exception as exc:
        return {
            "whale_score": 0,
            "whale_activity": "low",
            "whale_count": 0,
            "largest_transaction_btc": 0.0,
            "risk_level": "low",
            "reason": f"Failed to reach mempool.space API: {exc}",
        }, pd.DataFrame()

    # 2. fetch details & filter whales ────────────────────────────────────────
    whale_rows = []

    for txid in txids:
        tx = _get_tx_details(txid)
        if tx is None:
            continue

        value_btc = _tx_value_btc(tx)
        if value_btc < threshold_btc:
            continue

        whale_rows.append({
            "txid":        txid,
            "value_btc":   round(value_btc, 4),
            "fee_rate":    round(_fee_rate(tx), 2),
            "vin_count":   len(tx.get("vin", [])),
            "vout_count":  len(tx.get("vout", [])),
            "size_bytes":  tx.get("size", 0),
            "confirmed":   tx.get("status", {}).get("confirmed", False),
        })

    df = pd.DataFrame(whale_rows)

    # 3. compute metrics ──────────────────────────────────────────────────────
    whale_count = len(df)
    largest_btc = float(df["value_btc"].max()) if whale_count > 0 else 0.0

    # 4. score (0-100) ────────────────────────────────────────────────────────
    #   • count component  : each whale tx adds up to 10 pts (cap at 50)
    #   • size  component  : scaled by largest tx vs a "mega-whale" threshold
    count_score = min(whale_count * 10, 50)

    mega_threshold = 500.0          # BTC — considered a mega-whale
    size_score     = min((largest_btc / mega_threshold) * 50, 50)

    whale_score = int(count_score + size_score)

    # 5. activity label ───────────────────────────────────────────────────────
    if whale_score >= 75:
        whale_activity = "extreme"
    elif whale_score >= 50:
        whale_activity = "high"
    elif whale_score >= 25:
        whale_activity = "medium"
    else:
        whale_activity = "low"

    # 6. risk level ───────────────────────────────────────────────────────────
    if whale_score >= 70:
        risk_level = "high"
    elif whale_score >= 40:
        risk_level = "medium"
    else:
        risk_level = "low"

    # 7. reason ───────────────────────────────────────────────────────────────
    if whale_count == 0:
        reason = (
            f"No transactions exceeding {threshold_btc} BTC found in the "
            f"recent mempool sample ({TX_FETCH_LIMIT} txs inspected). "
            "Whale activity is low."
        )
    else:
        reason = (
            f"Detected {whale_count} transaction(s) above {threshold_btc} BTC "
            f"in the last {TX_FETCH_LIMIT} mempool entries. "
            f"Largest single transfer: {largest_btc:.2f} BTC. "
            f"Composite whale score: {whale_score}/100 → activity is {whale_activity.upper()}."
        )

    result = {
        "whale_score":             whale_score,
        "whale_activity":          whale_activity,
        "whale_count":             whale_count,
        "largest_transaction_btc": round(largest_btc, 4),
        "risk_level":              risk_level,
        "reason":                  reason,
    }

    return result, df
