"""
directional_probability_scorer.py
----------------------------------
Each module outputs its own UP probability (0-100%) directly.
Final probability = weighted average of active module probabilities.

This makes the result transparent:
  - Only Heatmap active with 27% upper liq → UP probability = 27%
  - Only Price/Volume active with OFI +0.10 → UP probability ≈ 68%
  - All modules → weighted average

Returns
-------
result : dict
    up_probability      – float, 0–100 %
    down_probability    – float, 0–100 %
    directional_signal  – str, "UP" | "DOWN" | "NEUTRAL"
    composite_score     – float, 0–100 (same as up_probability, for display)
    factors             – list[dict], each module with its probability and weight
    summary             – str
"""

from typing import Any


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────────────────────
# Module probability functions  (each returns P(UP) in 0–100)
# ─────────────────────────────────────────────────────────────────────────────

def _prob_heatmap(heatmap_result: dict) -> dict:
    """
    Heatmap: P(UP) = upper_share directly.
    If 27% of liquidation contracts are above price → 27% UP probability.
    Weight: 40%
    """
    upper_sh = float(heatmap_result.get("upper_share", 50))
    lower_sh = float(heatmap_result.get("lower_share", 50))
    total = upper_sh + lower_sh
    p_up = (upper_sh / total * 100) if total > 0 else 50.0
    p_up = _clamp(p_up, 5.0, 95.0)

    note = (
        f"Liquidation heatmap: {upper_sh:.1f}% of contracts above price, "
        f"{lower_sh:.1f}% below. "
        f"Direct probability → UP {p_up:.1f}%."
    )
    return {"name": "Liquidation Heatmap", "p_up": round(p_up, 1),
            "weight": 0.40, "annotation": note}


def _prob_price_volume(ai_result: dict) -> dict:
    """
    Price/Volume Module (OFI + IF volatility):
    Base: 50% ± shift from OFI ratio (each 0.01 OFI = 3% shift, capped ±40%).
    Amplified by volatility regime (ANOMALOUS ×1.3, ELEVATED ×1.15).
    Weight: 40%
    """
    ofi       = float(ai_result.get("ofi_ratio", 0.0))
    regime    = ai_result.get("volatility_regime", "NORMAL")
    count     = int(ai_result.get("recent_anomaly_count", 0))
    direction = ai_result.get("direction", "neutral").lower()

    # Base shift from OFI: 0.01 ratio = 3pp, capped at ±40pp
    ofi_shift = _clamp(ofi * 300, -40.0, 40.0)
    p_base = 50.0 + ofi_shift

    # Volatility amplifier: pulls further from 50%
    amp = {"ANOMALOUS": 1.30, "ELEVATED": 1.15, "NORMAL": 1.0}.get(regime, 1.0)
    p_up = 50.0 + (p_base - 50.0) * amp
    p_up = _clamp(p_up, 5.0, 95.0)

    note = (
        f"OFI ratio: {ofi:+.4f} → base shift {ofi_shift:+.1f}pp from 50%. "
        f"Volatility regime: {regime} (amplifier ×{amp:.2f}). "
        f"Direction: {direction.upper()}, anomaly flags: {count}/50. "
        f"→ UP {p_up:.1f}%."
    )
    return {"name": "Price / Volume Module (IF + OFI)", "p_up": round(p_up, 1),
            "weight": 0.40, "annotation": note}


def _prob_funding_rate(funding_result: dict) -> dict:
    """
    Funding Rate:
    Directly from raw rate value — every 0.01% = 5pp shift from 50%.
    Extreme sentiment overrides with fixed values (cascade/squeeze risk).
    Weight: 20%
    """
    sentiment = funding_result.get("funding_sentiment", "neutral")
    rate_pct  = float(funding_result.get("funding_rate_pct", 0.0))
    hist_mean = float(funding_result.get("hist_mean_pct", 0.0))

    if sentiment == "extreme_long":
        p_up = 30.0   # over-leveraged longs → liquidation cascade risk
    elif sentiment == "extreme_short":
        p_up = 70.0   # over-leveraged shorts → short squeeze risk
    else:
        # Linear: every 0.01% = 5pp from 50%, capped ±30pp
        shift = _clamp(rate_pct * 500, -30.0, 30.0)
        p_up = 50.0 + shift

    p_up = _clamp(p_up, 20.0, 80.0)

    note = (
        f"Funding rate: {rate_pct:+.4f}% ({sentiment}), "
        f"8-period avg {hist_mean:+.4f}%. "
        f"Linear shift: {rate_pct * 500:+.2f}pp from 50%. "
        f"→ UP {p_up:.1f}%."
    )
    return {"name": "Funding Rate", "p_up": round(p_up, 1),
            "weight": 0.20, "annotation": note}


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def compute_directional_probability(
    funding_result:   dict[str, Any],
    ai_result:        dict[str, Any],
    heatmap_result:   dict[str, Any],
    use_price_volume: bool = True,
    use_heatmap:      bool = True,
    use_funding:      bool = True,
) -> dict[str, Any]:
    """
    Final UP probability = weighted average of active module probabilities.
    Weights are renormalised automatically when modules are disabled.
    """

    all_modules = [
        (_prob_heatmap(heatmap_result),       use_heatmap),
        (_prob_price_volume(ai_result),        use_price_volume),
        (_prob_funding_rate(funding_result),   use_funding),
    ]
    factors = [m for m, active in all_modules if active]

    if not factors:
        return {
            "up_probability": 50.0, "down_probability": 50.0,
            "directional_signal": "NEUTRAL",
            "composite_score": 50.0,
            "factors": [], "summary": "No modules active.",
        }

    # Weighted average of module probabilities
    total_w = sum(f["weight"] for f in factors)
    up_prob = sum(f["p_up"] * f["weight"] for f in factors) / total_w
    up_prob = round(_clamp(up_prob, 0.0, 100.0), 1)
    down_prob = round(100.0 - up_prob, 1)

    # Normalise displayed weights
    for f in factors:
        f["weight"] = round(f["weight"] / total_w, 3)

    # Signal
    if up_prob >= 60:
        directional_signal = "UP"
    elif up_prob <= 40:
        directional_signal = "DOWN"
    else:
        directional_signal = "NEUTRAL"

    summary = (
        f"UP probability: {up_prob}% (weighted average of "
        f"{len(factors)} active module{'s' if len(factors) != 1 else ''}). "
        f"Signal: {directional_signal}."
    )

    return {
        "up_probability":     up_prob,
        "down_probability":   down_prob,
        "directional_signal": directional_signal,
        "composite_score":    up_prob,
        "factors":            factors,
        "summary":            summary,
    }
