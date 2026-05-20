import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

from funding_rate_detector import run_funding_rate_detector
from price_volume_detector import run_price_volume_detector
from liquidation_heatmap_proxy import run_liquidation_heatmap_proxy
from directional_probability_scorer import compute_directional_probability

st.set_page_config(
    page_title="Market Anomaly Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "configured" not in st.session_state:
    st.session_state.configured = False
    st.session_state.use_price_volume = True
    st.session_state.use_heatmap = True
    st.session_state.use_funding = True

# ── Welcome / configuration screen ───────────────────────────────────────────
if not st.session_state.configured:
    st.markdown("""
    <div style="max-width:520px;margin:60px auto 0;text-align:center">
        <h1 style="font-size:28px;font-weight:700;margin-bottom:6px">
            Market Anomaly Detection
        </h1>
        <p style="color:#888;font-size:15px;margin-bottom:40px">
            Real-Time BTC/USDT Futures Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("#### Select modules to include")
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

        use_pv = st.checkbox(
            "**Price / Volume Detector** (Isolation Forest + OFI)",
            value=st.session_state.use_price_volume,
            help="Detects anomalous candles via Isolation Forest and measures buying/selling pressure through Order Flow Imbalance."
        )
        use_hm = st.checkbox(
            "**Liquidation Heatmap Proxy**",
            value=st.session_state.use_heatmap,
            help="Estimates liquidation zones from open interest changes at different leverage levels."
        )
        use_fr = st.checkbox(
            "**Funding Rate Monitor**",
            value=st.session_state.use_funding,
            help="Tracks perpetual futures funding rate as a market positioning and stress indicator."
        )

        if not any([use_pv, use_hm, use_fr]):
            st.warning("Select at least one module.")
        else:
            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
            if st.button("▶  Start Monitoring", type="primary", width="stretch"):
                st.session_state.use_price_volume = use_pv
                st.session_state.use_heatmap = use_hm
                st.session_state.use_funding = use_fr
                st.session_state.configured = True
                st.rerun()

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        if st.button("Reset to defaults", width="stretch"):
            st.session_state.use_price_volume = True
            st.session_state.use_heatmap = True
            st.session_state.use_funding = True
            st.rerun()

    st.stop()

st_autorefresh(interval=60_000, key="refresh")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Page background */
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

/* Hide default streamlit header padding */
div[data-testid="stMetric"] label { font-size: 11px !important; color: #888 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 22px !important; }

/* Module cards */
.module-card {
    background: var(--background-color, #ffffff);
    border: 0.5px solid rgba(0,0,0,0.1);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    height: 100%;
}

/* Risk banner */
.banner-low    { background:#EAF3DE; color:#27500A; border-radius:10px; padding:.875rem 1.25rem; font-size:14px; border-left: 3px solid #639922; border-radius: 0 10px 10px 0;}
.banner-medium { background:#FAEEDA; color:#633806; border-radius:10px; padding:.875rem 1.25rem; font-size:14px; border-left: 3px solid #BA7517; border-radius: 0 10px 10px 0;}
.banner-high   { background:#FCEBEB; color:#501313; border-radius:10px; padding:.875rem 1.25rem; font-size:14px; border-left: 3px solid #E24B4A; border-radius: 0 10px 10px 0;}

/* Badges */
.badge { display:inline-block; font-size:11px; font-weight:600; padding:2px 8px; border-radius:4px; }
.badge-low  { background:#EAF3DE; color:#3B6D11; }
.badge-med  { background:#FAEEDA; color:#854F0B; }
.badge-high { background:#FCEBEB; color:#A32D2D; }
.badge-norm { background:#f0f0f0; color:#555; }

/* Section label */
.section-label { font-size:12px; font-weight:600; color:#888; text-transform:uppercase; letter-spacing:.06em; margin-bottom:.5rem; }

/* Module mini-stats */
.mini-stat-label { font-size:11px; color:#999; margin-bottom:2px; }
.mini-stat-value { font-size:18px; font-weight:500; }

/* Nav bar */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1rem;
    margin-bottom: 1.25rem;
    border-bottom: 0.5px solid rgba(0,0,0,0.08);
}
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #639922;
    margin-right: 5px;
}

/* Expander cleanup */
div[data-testid="stExpander"] { border: 0.5px solid rgba(0,0,0,0.08) !important; border-radius: 10px !important; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=55)
def load_data():
    funding_result, funding_df = run_funding_rate_detector(symbol="BTCUSDT")
    ai_result, ai_df = run_price_volume_detector(symbol="BTCUSDT", interval="1m", limit=500)
    heatmap_result, heatmap_source, active_levels, heatmap_zones, squeezed_levels = run_liquidation_heatmap_proxy(
        symbol="BTCUSDT",
        interval="5m",
        oi_period="5m",
        limit=500,
        leverages=[125, 100, 50],
        dispersion_factor=0.20,
        price_bin_size=50,
        storage_size=500,
        filter_peaks=False,
        signal_length=200,
        signal_mult=1.0
    )
    return funding_result, funding_df, ai_result, ai_df, heatmap_result, heatmap_source, active_levels, heatmap_zones, squeezed_levels

funding_result, funding_df, ai_result, ai_df, heatmap_result, heatmap_source, active_levels, heatmap_zones, squeezed_levels = load_data()

volatility_regime = ai_result.get("volatility_regime", "NORMAL")

# ── Reconfigure button ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Configuration")
    active = [m for m, on in [
        ("Price/Volume", st.session_state.use_price_volume),
        ("Heatmap", st.session_state.use_heatmap),
        ("Funding Rate", st.session_state.use_funding),
    ] if on]
    st.caption("Active modules: " + ", ".join(active))
    if st.button("Reconfigure modules", width="stretch"):
        st.session_state.configured = False
        st.rerun()


# ── Directional probability ──────────────────────────────────────────────────

dir_result = compute_directional_probability(
    funding_result=funding_result,
    ai_result=ai_result,
    heatmap_result=heatmap_result,
    use_price_volume=st.session_state.use_price_volume,
    use_heatmap=st.session_state.use_heatmap,
    use_funding=st.session_state.use_funding,
)

_dir_signal = dir_result["directional_signal"]   # "UP" | "DOWN" | "NEUTRAL"

# ── Helper: risk badge html ───────────────────────────────────────────────────

def risk_badge(level: str) -> str:
    cls = {"low": "badge-low", "medium": "badge-med", "high": "badge-high"}.get(level.lower(), "badge-norm")
    return f'<span class="badge {cls}">{level.upper()}</span>'


def regime_badge(regime: str) -> str:
    cls = {"NORMAL": "badge-norm", "ELEVATED": "badge-med", "ANOMALOUS": "badge-high"}.get(regime, "badge-norm")
    return f'<span class="badge {cls}">{regime}</span>'


def attraction_badge(attr: str) -> str:
    cls = {"up": "badge-high", "down": "badge-med", "neutral": "badge-norm"}.get(attr.lower(), "badge-norm")
    arrow = {"up": "↑", "down": "↓", "neutral": "→"}.get(attr.lower(), "")
    return f'<span class="badge {cls}">{arrow} {attr.upper()}</span>'


# ── Nav bar ───────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="nav-bar">'
    '<span style="font-size:17px;font-weight:500">Market anomaly detection</span>'
    '<span style="font-size:12px;color:#999"><span class="live-dot"></span>Live · auto-refresh 60s</span>'
    '</div>',
    unsafe_allow_html=True
)


# ── Risk banner ───────────────────────────────────────────────────────────────

# ── Overview ─────────────────────────────────────────────────────────────────

_up_p  = dir_result["up_probability"]
_dn_p  = dir_result["down_probability"]
_sig_color  = {"UP": "#27A570", "DOWN": "#E24B4A", "NEUTRAL": "#888"}.get(_dir_signal, "#888")

st.markdown('<p class="section-label">Overview</p>', unsafe_allow_html=True)

ov1, ov2, ov3 = st.columns(3)

ov1.metric("BTC Price", f"${heatmap_result['current_price']:,.1f}")

ov2.markdown(
    f'<div style="padding:4px 0">'
    f'<div style="font-size:11px;color:#888;margin-bottom:6px">Signal</div>'
    f'<div style="font-size:22px;font-weight:700;color:{_sig_color};line-height:1">{_dir_signal}</div>'
    f'<div style="font-size:12px;color:#aaa;margin-top:4px">▲ {_up_p}% · ▼ {_dn_p}%</div>'
    f'</div>',
    unsafe_allow_html=True
)

ov3.metric("Main liquidity zone", f"${heatmap_result['main_zone_price']:,.2f}")

st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)


# ── Three module cards ────────────────────────────────────────────────────────

st.markdown('<p class="section-label">Module signals</p>', unsafe_allow_html=True)

_fr_sent       = funding_result["funding_sentiment"]
_fr_sent_color = {
    "extreme_long": "#E24B4A", "bullish": "#27A570",
    "neutral": "#888",
    "bearish": "#E24B4A", "extreme_short": "#27A570",
}.get(_fr_sent, "#888")
_fr_sent_bg = {
    "extreme_long": "#FCEBEB", "bullish": "#EAF3DE",
    "neutral": "#f0f0f0",
    "bearish": "#FCEBEB", "extreme_short": "#EAF3DE",
}.get(_fr_sent, "#f0f0f0")

def _card_funding():
    st.markdown(
        f"""<div class="module-card">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      border-bottom:0.5px solid rgba(0,0,0,0.07);padding-bottom:.75rem;margin-bottom:.875rem">
            <span style="font-size:13px;font-weight:500">Funding rate</span>
            <span style="font-size:11px;font-weight:600;padding:2px 8px;border-radius:4px;
                         background:{_fr_sent_bg};color:{_fr_sent_color}">
              {_fr_sent.upper().replace("_", " ")}
            </span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:.875rem">
            <div><div class="mini-stat-label">Rate</div>
              <div class="mini-stat-value">{funding_result['funding_rate_pct']:+.4f}%</div></div>
            <div><div class="mini-stat-label">8-period avg</div>
              <div class="mini-stat-value">{funding_result['hist_mean_pct']:+.4f}%</div></div>
            <div><div class="mini-stat-label">Next funding</div>
              <div class="mini-stat-value">{funding_result['next_funding_time']}</div></div>
          </div>
          <div style="font-size:12px;color:#888;border-top:0.5px solid rgba(0,0,0,0.07);padding-top:.75rem">
            Direction bias: {funding_result['direction_bias'].upper()} · Score: {funding_result['funding_score']}/100
          </div></div>""",
        unsafe_allow_html=True
    )

def _card_price_volume():
    latest_vol = ai_result.get("latest_volatility", "N/A")
    st.markdown(
        f"""<div class="module-card">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      border-bottom:0.5px solid rgba(0,0,0,0.07);padding-bottom:.75rem;margin-bottom:.875rem">
            <span style="font-size:13px;font-weight:500">Volatility regime</span>
            {regime_badge(volatility_regime)}
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:.875rem">
            <div><div class="mini-stat-label">Anomaly score</div>
              <div class="mini-stat-value">{ai_result['anomaly_score']}</div></div>
            <div><div class="mini-stat-label">Recent flags</div>
              <div class="mini-stat-value">{ai_result['recent_anomaly_count']}</div></div>
            <div><div class="mini-stat-label">OFI Direction</div>
              <div class="mini-stat-value">{ai_result['direction'].upper()}</div></div>
          </div>
          <div style="font-size:12px;color:#888;border-top:0.5px solid rgba(0,0,0,0.07);padding-top:.75rem">
            OFI: {ai_result.get('ofi_ratio', 0):+.4f} · Volatility: {latest_vol}
          </div></div>""",
        unsafe_allow_html=True
    )

def _card_heatmap():
    st.markdown(
        f"""<div class="module-card">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      border-bottom:0.5px solid rgba(0,0,0,0.07);padding-bottom:.75rem;margin-bottom:.875rem">
            <span style="font-size:13px;font-weight:500">Liquidation heatmap</span>
            {attraction_badge(heatmap_result['liquidity_attraction'])}
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:.875rem">
            <div><div class="mini-stat-label">Lower liq.</div>
              <div class="mini-stat-value">{heatmap_result['lower_share']}%</div></div>
            <div><div class="mini-stat-label">Upper liq.</div>
              <div class="mini-stat-value">{heatmap_result['upper_share']}%</div></div>
            <div><div class="mini-stat-label">Zone strength</div>
              <div class="mini-stat-value">{heatmap_result['main_zone_strength']}%</div></div>
          </div>
          <div style="font-size:12px;color:#888;border-top:0.5px solid rgba(0,0,0,0.07);padding-top:.75rem">
            Main zone at ${heatmap_result['main_zone_price']:,.2f} · Dominant side: {heatmap_result['dominant_liquidity_side']}
          </div></div>""",
        unsafe_allow_html=True
    )

_active_cards = [
    (st.session_state.use_funding,       _card_funding),
    (st.session_state.use_price_volume,  _card_price_volume),
    (st.session_state.use_heatmap,       _card_heatmap),
]
_visible = [(fn) for active, fn in _active_cards if active]
if _visible:
    _cols = st.columns(len(_visible))
    for _col, _fn in zip(_cols, _visible):
        with _col:
            _fn()

st.markdown("<div style='margin-bottom:1.75rem'></div>", unsafe_allow_html=True)


# ── Directional probability block ────────────────────────────────────────────

st.markdown('<p class="section-label">Directional probability</p>', unsafe_allow_html=True)

up_p   = _up_p
dn_p   = _dn_p
sig    = _dir_signal
cscore = dir_result["composite_score"]

up_col  = "#27A570" if up_p > dn_p else "#aaa"
dn_col  = "#E24B4A" if dn_p > up_p else "#aaa"
sig_col = _sig_color

st.markdown(f"""
<div style="border:0.5px solid rgba(0,0,0,0.1);border-radius:12px;
            padding:1.25rem 1.5rem;background:var(--background-color,#fff);margin-bottom:1rem">

  <div style="font-size:13px;font-weight:500;border-bottom:0.5px solid rgba(0,0,0,0.07);
              padding-bottom:.875rem;margin-bottom:1rem">
    Movement probability estimate
  </div>

  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px">
    <span style="font-size:13px;color:#666;font-weight:500">&#9650; UP</span>
    <span style="font-size:24px;font-weight:700;color:{up_col}">{up_p}%</span>
  </div>
  <div style="background:#f0f0f0;border-radius:6px;height:12px;overflow:hidden;margin-bottom:16px">
    <div style="background:{up_col};width:{up_p}%;height:100%;border-radius:6px"></div>
  </div>

  <div style="text-align:center;font-size:11px;color:#ccc;letter-spacing:.1em;margin-bottom:16px">── VS ──</div>

  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px">
    <span style="font-size:13px;color:#666;font-weight:500">&#9660; DOWN</span>
    <span style="font-size:24px;font-weight:700;color:{dn_col}">{dn_p}%</span>
  </div>
  <div style="background:#f0f0f0;border-radius:6px;height:12px;overflow:hidden;margin-bottom:18px">
    <div style="background:{dn_col};width:{dn_p}%;height:100%;border-radius:6px"></div>
  </div>

  <div style="display:flex;gap:32px;border-top:0.5px solid rgba(0,0,0,0.07);padding-top:.875rem">
    <div>
      <div style="font-size:11px;color:#999;margin-bottom:3px">Composite score</div>
      <div style="font-size:20px;font-weight:600">{cscore:+.1f}&nbsp;/&nbsp;100</div>
    </div>
    <div>
      <div style="font-size:11px;color:#999;margin-bottom:3px">Signal</div>
      <div style="font-size:20px;font-weight:700;color:{sig_col}">{sig}</div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:.5rem'></div>", unsafe_allow_html=True)


# ── Heatmap chart ─────────────────────────────────────────────────────────────

st.markdown('<p class="section-label">Liquidation zones chart</p>', unsafe_allow_html=True)

if heatmap_source.empty or heatmap_zones.empty:
    st.warning("Not enough data to draw liquidation heatmap.")
else:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=heatmap_source["close_time"],
        open=heatmap_source["open"],
        high=heatmap_source["high"],
        low=heatmap_source["low"],
        close=heatmap_source["close"],
        name="BTCUSDT",
        increasing_line_color="#1D9E75",
        decreasing_line_color="#E24B4A",
    ))

    recent_start = heatmap_source["close_time"].iloc[0]
    recent_end   = heatmap_source["close_time"].iloc[-1]
    current_price = heatmap_result["current_price"]

    price_range = current_price * 0.08
    top_zones = heatmap_zones[
        (heatmap_zones["zone_price"] >= current_price - price_range) &
        (heatmap_zones["zone_price"] <= current_price + price_range)
    ].sort_values("zone_strength", ascending=False).head(30)
    max_strength = top_zones["zone_strength"].max()

    for _, zone in top_zones.iterrows():
        lw = 1 + (zone["zone_strength"] / max_strength) * 4 if max_strength > 0 else 2
        color = "rgba(40,110,255,0.40)" if zone["side"] == "long_liquidation" else "rgba(120,200,0,0.40)"
        fig.add_trace(go.Scatter(
            x=[recent_start, recent_end],
            y=[zone["zone_price"], zone["zone_price"]],
            mode="lines",
            line=dict(color=color, width=lw),
            hovertemplate=(
                f"Price: %{{y}}<br>Type: {zone['side']}<br>"
                f"Contracts: {zone['contracts']:.2f}<br>"
                f"Strength: {zone['zone_strength']:.2f}%<extra></extra>"
            ),
            showlegend=False
        ))

    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="rgba(100,100,100,0.5)",
        line_width=1,
        annotation_text=f"  Current ${current_price:,.1f}",
        annotation_position="top left",
        annotation_font_size=11,
    )

    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=10, r=10, t=20, b=20),
        yaxis_title="Price (USDT)",
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        font=dict(size=11),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.04)", gridwidth=0.5)

    st.plotly_chart(fig, width="stretch")

st.markdown("<div style='margin-bottom:.5rem'></div>", unsafe_allow_html=True)


# ── Details (expanders) ───────────────────────────────────────────────────────

st.markdown('<p class="section-label">Details</p>', unsafe_allow_html=True)

with st.expander(f"Funding rate history ({len(funding_df)} records)"):
    st.caption(funding_result["reason"])
    if funding_df.empty:
        st.info("No funding rate history available.")
    else:
        st.dataframe(funding_df, width="stretch", hide_index=True)

with st.expander(f"Isolation Forest candle data ({len(ai_df)} rows)"):
    st.caption(ai_result["reason"])
    cols_show = ["close_time","close","volume","price_return","candle_range",
                 "volume_z_score","volatility","raw_score","is_anomaly","anomaly_strength"]
    st.dataframe(ai_df[cols_show].tail(100), width="stretch", hide_index=True)

with st.expander("Top liquidation zones"):
    if heatmap_zones.empty:
        st.warning("No estimated liquidation zones found.")
    else:
        st.dataframe(
            heatmap_zones.sort_values("impact_score", ascending=False).head(10),
            width="stretch", hide_index=True
        )

with st.expander("Recently squeezed levels"):
    if squeezed_levels.empty:
        st.info("No recently squeezed liquidation levels.")
    else:
        st.dataframe(squeezed_levels.tail(30), width="stretch", hide_index=True)

with st.expander("Active liquidation levels"):
    if active_levels.empty:
        st.warning("No active estimated liquidation levels.")
    else:
        st.dataframe(active_levels.tail(100), width="stretch", hide_index=True)

with st.expander("Source data (futures price + open interest)"):
    if heatmap_source.empty:
        st.warning("No source data.")
    else:
        st.dataframe(heatmap_source.tail(100), width="stretch", hide_index=True)

with st.expander("Methodology"):
    st.markdown("""
**1. Funding Rate Module**
Fetches the current BTC perpetual futures funding rate from Binance (/fapi/v1/premiumIndex).
Positive rate = longs pay shorts (bullish sentiment). Negative = shorts pay longs (bearish).
Extreme values (> ±0.1%) signal over-leveraged positioning and liquidation/squeeze risk.
Each 0.01% of rate shifts UP probability ±5pp from 50%. Extreme sentiment overrides apply:
extreme long → P(UP) = 30%, extreme short → P(UP) = 70%. Weight: 20%.

**2. Price/Volume Module (Isolation Forest + OFI)**
Trains an unsupervised Isolation Forest on 500 recent 1-minute BTCUSDT candles
using 9 features: price return, candle range, close position, volume change,
volume z-score, taker buy ratio, volatility, momentum_3, momentum_10.
Classifies the volatility regime as NORMAL / ELEVATED / ANOMALOUS.
Direction bias is derived from Order Flow Imbalance (OFI = taker buy − taker sell volume)
over the last 50 candles, normalised by total volume. OFI ratio > 0.04 = bullish, < −0.04 = bearish.
Each 0.01 OFI shift moves UP probability ±3pp from 50%, amplified by volatility regime. Weight: 40%.

**3. Liquidation Heatmap Proxy**
Estimates potential liquidation zones using Binance futures 5-minute OHLCV data and open
interest changes. Projects liquidation thresholds for leverage levels 125×, 100×, 50×
above and below the current price. UP probability = share of estimated contracts above price.
If 27% of contracts sit above price → P(UP) = 27%. Weight: 40%.

**Directional Probability Scorer**
Each active module outputs its own UP probability (0–100%) directly.
Final UP probability = weighted average of active module probabilities.
Weights are renormalised automatically when modules are disabled.
Signal: UP probability ≥ 60% → UP · ≤ 40% → DOWN · otherwise → NEUTRAL.

*This system uses public market data only. Liquidation zones are estimated.
Not a trading recommendation.*
    """)

# Footer
st.markdown(
    "<div style='margin-top:2rem;font-size:11px;color:#bbb;text-align:center'>"
    "AI-Based Market Anomaly Detection · public data only · not financial advice"
    "</div>",
    unsafe_allow_html=True
)