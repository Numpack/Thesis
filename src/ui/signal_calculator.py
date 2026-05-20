"""
signal_calculator.py
--------------------
Combines three independent market modules into transparent upside / downside /
neutral percentages.

Important: the output is not a statistical probability learned from labels.
It is an interpretable confidence score based on weighted evidence from:
1) liquidation heatmap,
2) AI volatility regime,
3) whale activity.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise(up: float, down: float, neutral: float) -> Dict[str, float]:
    up = max(0.0, up)
    down = max(0.0, down)
    neutral = max(0.0, neutral)
    total = up + down + neutral

    if total <= 0:
        return {"up": 33.33, "down": 33.33, "neutral": 33.34}

    return {
        "up": round(up / total * 100, 2),
        "down": round(down / total * 100, 2),
        "neutral": round(neutral / total * 100, 2),
    }


def _ai_direction(raw_direction: str) -> str:
    direction = str(raw_direction or "neutral").lower()

    if direction in ["up", "bullish", "long"]:
        return "up"
    if direction in ["down", "bearish", "short"]:
        return "down"
    return "neutral"


def calculate_market_probabilities(
    whale_result: Dict[str, Any],
    ai_result: Dict[str, Any],
    heatmap_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an explainable market signal.

    Output fields:
    - up_probability
    - down_probability
    - neutral_probability
    - signal
    - risk
    - annotations
    - contribution_rows
    """

    # Neutral baseline. Direction must be earned by evidence.
    up_score = 25.0
    down_score = 25.0
    neutral_score = 50.0

    annotations: List[str] = []
    contribution_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Liquidation heatmap direction
    # ------------------------------------------------------------------
    upper_share = _safe_float(heatmap_result.get("upper_share"))
    lower_share = _safe_float(heatmap_result.get("lower_share"))
    heatmap_imbalance = upper_share - lower_share
    heatmap_strength = abs(heatmap_imbalance)
    liquidity_attraction = str(heatmap_result.get("liquidity_attraction", "neutral")).lower()

    heatmap_add = _clamp(heatmap_strength * 0.55, 0.0, 30.0)
    neutral_reduction = _clamp(heatmap_strength * 0.30, 0.0, 18.0)

    if heatmap_imbalance > 0:
        up_score += heatmap_add
        neutral_score -= neutral_reduction
        heatmap_note = (
            f"Heatmap: upper liquidity is {upper_share:.2f}% vs lower liquidity {lower_share:.2f}%. "
            f"This adds upside pressure, but only according to imbalance strength."
        )
    elif heatmap_imbalance < 0:
        down_score += heatmap_add
        neutral_score -= neutral_reduction
        heatmap_note = (
            f"Heatmap: lower liquidity is {lower_share:.2f}% vs upper liquidity {upper_share:.2f}%. "
            f"This adds downside pressure, but only according to imbalance strength."
        )
    else:
        heatmap_note = "Heatmap: upper and lower liquidity are balanced, so it adds no directional pressure."

    annotations.append(heatmap_note)
    contribution_rows.append({
        "Module": "Liquidation heatmap",
        "Input": f"Upper {upper_share:.2f}% / Lower {lower_share:.2f}%",
        "Effect": "Upside" if heatmap_imbalance > 0 else "Downside" if heatmap_imbalance < 0 else "Neutral",
        "Explanation": heatmap_note,
    })

    # ------------------------------------------------------------------
    # 2. AI volatility regime
    # ------------------------------------------------------------------
    regime = str(ai_result.get("volatility_regime", "NORMAL")).upper()
    ai_dir = _ai_direction(ai_result.get("direction", "neutral"))
    anomaly_score = _safe_float(ai_result.get("anomaly_score"))
    recent_flags = int(_safe_float(ai_result.get("recent_anomaly_count")))

    if regime == "NORMAL":
        neutral_score += 8.0
        ai_effect = "Neutral"
        ai_note = (
            "AI volatility regime is NORMAL. The AI module is treated as a risk filter, "
            "so it does not add bullish or bearish pressure in normal conditions."
        )
    elif regime == "ELEVATED":
        neutral_score -= 8.0
        if ai_dir == "up":
            up_score += 8.0
            ai_effect = "Upside"
        elif ai_dir == "down":
            down_score += 8.0
            ai_effect = "Downside"
        else:
            up_score += 4.0
            down_score += 4.0
            ai_effect = "Movement risk"
        ai_note = (
            f"AI volatility regime is ELEVATED with anomaly score {anomaly_score:.2f} "
            f"and {recent_flags} recent anomaly flag(s). It increases movement risk."
        )
    else:
        neutral_score -= 15.0
        if ai_dir == "up":
            up_score += 15.0
            ai_effect = "Upside"
        elif ai_dir == "down":
            down_score += 15.0
            ai_effect = "Downside"
        else:
            up_score += 7.5
            down_score += 7.5
            ai_effect = "Movement risk"
        ai_note = (
            f"AI volatility regime is ANOMALOUS with anomaly score {anomaly_score:.2f}. "
            "This means the market state is statistically unusual."
        )

    annotations.append(ai_note)
    contribution_rows.append({
        "Module": "AI volatility regime",
        "Input": f"{regime}, score {anomaly_score:.2f}, recent flags {recent_flags}",
        "Effect": ai_effect,
        "Explanation": ai_note,
    })

    # ------------------------------------------------------------------
    # 3. Whale activity
    # ------------------------------------------------------------------
    whale_score = _safe_float(whale_result.get("whale_score"))
    whale_activity = str(whale_result.get("whale_activity", "low")).upper()

    if whale_score < 50:
        neutral_score += 5.0
        whale_effect = "Neutral"
        whale_note = (
            f"Whale score is {whale_score:.0f} ({whale_activity}). "
            "There is no strong on-chain confirmation from large BTC transfers."
        )
    else:
        movement_boost = _clamp(whale_score / 8.0, 6.0, 15.0)
        neutral_score -= movement_boost

        if liquidity_attraction == "up":
            up_score += movement_boost
            whale_effect = "Confirms upside"
        elif liquidity_attraction == "down":
            down_score += movement_boost
            whale_effect = "Confirms downside"
        else:
            up_score += movement_boost / 2.0
            down_score += movement_boost / 2.0
            whale_effect = "Movement risk"

        whale_note = (
            f"Whale score is {whale_score:.0f} ({whale_activity}). "
            "Large BTC transfers increase movement risk, but direction is taken from market context."
        )

    annotations.append(whale_note)
    contribution_rows.append({
        "Module": "Whale activity",
        "Input": f"Score {whale_score:.0f}, activity {whale_activity}",
        "Effect": whale_effect,
        "Explanation": whale_note,
    })

    # ------------------------------------------------------------------
    # Final probabilities and signal
    # ------------------------------------------------------------------
    probs = _normalise(up_score, down_score, neutral_score)
    up_p = probs["up"]
    down_p = probs["down"]
    neutral_p = probs["neutral"]

    directional_max = max(up_p, down_p)
    directional_gap = abs(up_p - down_p)

    if neutral_p >= directional_max or directional_gap < 7:
        signal = "neutral"
    else:
        signal = "up" if up_p > down_p else "down"

    if signal == "neutral":
        risk = "low"
    elif directional_max >= 60 and neutral_p <= 25:
        risk = "high"
    elif directional_max >= 45 or regime in ["ELEVATED", "ANOMALOUS"] or whale_score >= 50:
        risk = "medium"
    else:
        risk = "low"

    final_note = (
        f"Final score: Up {up_p:.2f}%, Down {down_p:.2f}%, Neutral {neutral_p:.2f}%. "
        f"The final signal is {signal.upper()} because it has the strongest confirmed evidence "
        "after combining heatmap, AI regime and whale activity."
    )
    annotations.append(final_note)

    return {
        "up_probability": up_p,
        "down_probability": down_p,
        "neutral_probability": neutral_p,
        "signal": signal,
        "risk": risk,
        "annotations": annotations,
        "contribution_rows": contribution_rows,
        "heatmap_imbalance": round(heatmap_imbalance, 2),
    }
