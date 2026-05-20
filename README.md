# AI-Based Real-Time Detection of Market Anomalies and Suspicious Activities

Bachelor's Thesis — Neapolis University Paphos  
Author: Danila  
Supervisor: Dr. Marios Touloupou

---

## Overview

This is a real-time price anomaly detection dashboard for BTCUSDT perpetual futures trading data on Binance. Using machine learning and microstructure signals available on the trading side, this tool will detect anomalies that represent suspicious buying/selling price/volume behaviour and will output a probability of price moving in a direction.

---

## Modules

| Module | Description |
|---|---|
| `price_volume_detector.py` | Isolation Forest is an unsupervised anomaly detection algorithm that is efficient for high-dimensional datasets. The Isolation Forest is trained on the 9 features describing the market condition. It identifies anomalous candles (duration of time when price is relatively steady, followed by a market burst) and determines the volatility regime (NORMAL / ELEVATED / ANOMALOUS). |
| `funding_rate_detector.py` | Monitors Binance perpetual funding rate. Flags extreme values (> ±0.1%) as potential reversal signals |
| `liquidation_heatmap_proxy.py` | Estimates liquidation risk levels using open interest delta and leverage projections (50×, 100×, 125×) |
| `directional_probability_scorer.py` | Combines all module outputs into a single P(UP) / P(DOWN) signal using weighted aggregation |
| `app.py` | Streamlit dashboard. Auto-refreshes every 60 seconds |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/thesis5.git
cd thesis5
```

**2. Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the dashboard**
```bash
streamlit run src/ui/app.py
```

The dashboard will open at `http://localhost:8501`

---

## Data Source

All data is fetched in real time from the **Binance Futures public REST API** (no API key required).

- Candle data: `https://fapi.binance.com/fapi/v1/klines`
- Funding rate: `https://fapi.binance.com/fapi/v1/premiumIndex`
- Open interest: `https://fapi.binance.com/futures/data/openInterestHist`

---

## System Architecture

```
Binance API
    │
    ├── price_volume_detector     → anomaly score, volatility regime, OFI direction
    ├── funding_rate_detector     → funding rate signal, extreme flag
    ├── liquidation_heatmap_proxy → UP/DOWN bias, liquidation levels
    │
    └── directional_probability_scorer → P(UP) / P(DOWN)
                                              │
                                         Streamlit Dashboard (app.py)
```

---

## Key Parameters

| Parameter | Value | Reason |
|---|---|---|
| Candles fetched | 500 | Balances recency and model training stability |
| Isolation Forest trees | 200 | Reduces variance in anomaly scores |
| Contamination | 0.05 | 5% expected anomaly rate (Liu et al., 2008) |
| Recent window | 100 candles | Smooths direction signal, reduces noise |
| OFI threshold | ±0.04 | Empirically determined threshold for directional bias |
| Funding rate threshold | ±0.1% | Extreme level per Binance perpetual contract mechanics |
| Refresh interval | 60 seconds | Balances real-time responsiveness with API rate limits |
