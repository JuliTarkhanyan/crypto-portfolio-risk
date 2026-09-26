# ============================================================
# APP.PY
# CRYPTO PORTFOLIO & RISK ANALYSIS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# GLOBAL UI / VISUAL THEME
# ============================================================

st.markdown(
    """
<style>
:root {
    --bg: #0b1020;
    --panel: #111827;
    --panel-2: #151d2f;
    --border: rgba(148, 163, 184, 0.16);
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #7c9cff;
    --accent-2: #9b8cff;
    --positive: #34d399;
    --negative: #fb7185;
    --warning: #fbbf24;
}
/* Main application background */
.stApp {
    background:
        radial-gradient(
            circle at 8% 0%,
            rgba(124,156,255,0.09),
            transparent 28%
        ),
        radial-gradient(
            circle at 92% 8%,
            rgba(155,140,255,0.07),
            transparent 24%
        ),
        var(--bg);
}
/* Main content width */
.main .block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(17,24,39,0.98) 0%,
            rgba(10,15,28,0.98) 100%
        );
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
}
/* Hero header */
.crypto-hero {
    position: relative;
    overflow: hidden;
    padding: 1.65rem 1.8rem;
    margin-bottom: 1.4rem;
    border: 1px solid var(--border);
    border-radius: 22px;
    background:
        linear-gradient(
            135deg,
            rgba(124,156,255,0.14),
            rgba(17,24,39,0.72) 48%,
            rgba(155,140,255,0.10)
        );
    box-shadow:
        0 18px 55px rgba(0,0,0,0.20);
}
.hero-kicker {
    color: var(--accent);
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.hero-title {
    color: var(--text);
    font-size: 2.15rem;
    line-height: 1.1;
    font-weight: 800;
    margin: 0;
}
.hero-subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.55rem;
    max-width: 850px;
}
.hero-status {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    margin-top: 1rem;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(52,211,153,0.22);
    background: rgba(52,211,153,0.08);
    color: #86efac;
    font-size: 0.76rem;
    font-weight: 700;
}
.hero-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--positive);
    box-shadow:
        0 0 12px rgba(52,211,153,0.75);
}
/* Section labels */
.section-label {
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 1.4rem;
    margin-bottom: 0.3rem;
}
/* KPI cards */
div[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(21,29,47,0.96),
            rgba(15,23,42,0.92)
        );
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow:
        0 10px 30px rgba(0,0,0,0.13);
    min-height: 105px;
}
div[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricDelta"] {
    font-weight: 700 !important;
}
/* Buttons */
.stButton > button {
    border: 1px solid rgba(124,156,255,0.28);
    border-radius: 11px;
    background:
        linear-gradient(
            135deg,
            rgba(124,156,255,0.18),
            rgba(155,140,255,0.12)
        );
    color: #eef2ff;
    font-weight: 750;
    min-height: 42px;
    transition: all 0.18s ease;
}
.stButton > button:hover {
    border-color: rgba(124,156,255,0.62);
    background:
        linear-gradient(
            135deg,
            rgba(124,156,255,0.27),
            rgba(155,140,255,0.20)
        );
    transform: translateY(-1px);
}
/* Tabs */
button[data-baseweb="tab"] {
    color: #94a3b8;
    font-weight: 700;
    border-radius: 10px 10px 0 0;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #eef2ff;
}
div[data-baseweb="tab-highlight"] {
    background: var(--accent);
    height: 3px;
    border-radius: 3px;
}
/* Inputs */
div[data-baseweb="select"] > div,
div[data-testid="stDateInput"] input,
div[data-testid="stNumberInput"] input {
    background: rgba(15,23,42,0.72);
    border-color: var(--border);
    border-radius: 10px;
}
/* Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}
/* Expanders */
details[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: rgba(17,24,39,0.58);
    margin-bottom: 0.65rem;
}
details[data-testid="stExpander"] summary {
    font-weight: 750;
}
/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 13px;
    border: 1px solid var(--border);
}
/* Plotly charts */
div[data-testid="stPlotlyChart"] {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.25rem;
    background: rgba(15,23,42,0.34);
    overflow: hidden;
}
/* Headings */
h1, h2, h3 {
    color: var(--text) !important;
    letter-spacing: -0.02em;
}
/* Divider */
hr {
    border-color: var(--border) !important;
}
/* Footer */
.app-footer {
    margin-top: 2rem;
    padding: 1rem 0 0.5rem;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.75rem;
    text-align: center;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data import (
    download_top_coins,
    download_top_100_historical_prices,
    download_coin_details_by_id,
    download_market_chart_by_id,
)

from src.returns import (
    calculate_daily_returns,
    calculate_log_returns,
    annualized_return,
    annualized_volatility,
)

from src.risk import (
    portfolio_returns,
    covariance_matrix,
    correlation_matrix,
    portfolio_variance,
    portfolio_volatility,
    historical_var,
    historical_cvar,
    parametric_var,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
    downside_deviation,
    beta,
)

from src.optimization import (
    calculate_weights,
)

from src.backtest import (
    buy_and_hold,
    periodic_rebalance,
    backtest_metrics,
)

from src.monte_carlo import (
    monte_carlo_simulation,
)

from src.crypto_research import (
    get_coin_research,
)

from src.single_crypto_analysis import (
    render_single_crypto_research,
)

from src.ai_analysis import (
    generate_ai_analysis,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crypto Portfolio & Risk Analysis",
    page_icon="₿",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
<div class="crypto-hero">
    <div class="hero-kicker">
        Portfolio Intelligence Platform
    </div>
    <div class="hero-title">
        ₿ Crypto Portfolio Analytics
    </div>
    <div class="hero-subtitle">
        Portfolio construction · Risk analysis · Performance ·
        Backtesting · Monte Carlo simulation · Optimization · AI insights
    </div>
    <div class="hero-status">
        <span class="hero-dot"></span>
        Market analytics workspace
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD TOP 100 COINS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_top_coins():

    return download_top_coins(
        limit=100
    )


try:

    top_coins = load_top_coins()

except Exception as e:

    st.error(
        f"Could not load cryptocurrency market data: {e}"
    )

    st.stop()


# ============================================================
# NORMALIZE TOP COINS DATA
# ============================================================

column_mapping = {

    "id": "CoinGecko ID",

    "symbol": "Symbol",

    "name": "Name",

    "market_cap_rank": "Rank",

    "current_price": "Price",

    "market_cap": "Market Cap",

    "total_volume": "24h Volume",

    "price_change_percentage_24h": "24h Change",

    "price_change_percentage_7d_in_currency":
        "7d Change",

    "price_change_percentage_30d_in_currency":
        "30d Change",

}


top_coins = top_coins.rename(
    columns=column_mapping
)


# ============================================================
# MAKE SURE REQUIRED COLUMNS EXIST
# ============================================================

required_columns = [
    "CoinGecko ID",
    "Symbol",
    "Name",
    "Rank",
    "Price",
    "Market Cap",
    "24h Change",
]


for column in required_columns:

    if column not in top_coins.columns:

        top_coins[column] = np.nan


# ============================================================
# CLEAN DATA
# ============================================================

top_coins["Rank"] = pd.to_numeric(
    top_coins["Rank"],
    errors="coerce",
)

top_coins["Symbol"] = (
    top_coins["Symbol"]
    .fillna("")
    .astype(str)
    .str.upper()
)

top_coins["Name"] = (
    top_coins["Name"]
    .fillna("")
    .astype(str)
)

top_coins["CoinGecko ID"] = (
    top_coins["CoinGecko ID"]
    .fillna("")
    .astype(str)
)


top_coins = (
    top_coins
    .dropna(subset=["CoinGecko ID"])
    .sort_values("Rank")
    .reset_index(drop=True)
)


# ============================================================
# PORTFOLIO LABEL
# ============================================================

top_coins["Portfolio Label"] = (

    "#"
    + top_coins["Rank"]
        .fillna(0)
        .astype(int)
        .astype(str)

    + " — "

    + top_coins["Name"]

    + " ("

    + top_coins["Symbol"]

    + ")"

)


# ============================================================
# COIN OPTIONS
# ============================================================

crypto_options = dict(

    zip(
        top_coins["Portfolio Label"],
        top_coins["CoinGecko ID"],
    )

)


# ============================================================
# HELPER: COIN NAME
# ============================================================

def coin_name(coin_id):

    row = top_coins[
        top_coins["CoinGecko ID"]
        == coin_id
    ]


    if row.empty:

        return coin_id


    return row.iloc[0]["Name"]


# ============================================================
# SIDEBAR — PORTFOLIO CONTROL CENTER
# ============================================================

st.sidebar.markdown(
    """
<div style="
    padding: 0.4rem 0.2rem 1.2rem 0.2rem;
">
    <div style="
        color:#7c9cff;
        font-size:0.68rem;
        font-weight:800;
        letter-spacing:0.18em;
        text-transform:uppercase;
        margin-bottom:0.35rem;
    ">
        CRYPTO ANALYTICS
    </div>
    <div style="
        color:#f8fafc;
        font-size:1.35rem;
        font-weight:800;
        line-height:1.15;
    ">
        Portfolio Control
    </div>
    <div style="
        color:#64748b;
        font-size:0.76rem;
        margin-top:0.45rem;
        line-height:1.45;
    ">
        Configure your portfolio,
        analysis period and weighting model.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")


# ============================================================
# ANALYSIS PERIOD
# ============================================================

st.sidebar.markdown(
    """
<div style="
    color:#94a3b8;
    font-size:0.70rem;
    font-weight:800;
    letter-spacing:0.13em;
    text-transform:uppercase;
    margin-bottom:0.65rem;
">
    📅 Analysis Period
</div>
""",
    unsafe_allow_html=True,
)

# These bounds were referenced but never defined, which is what caused
# the NameError. max_available_date/min_available_date bound the picker;
# default_start_date/default_end_date set its initial value. Adjust the
# lookback window (365 days here) to whatever your data provider supports.
max_available_date = pd.Timestamp.now().normalize().date()
min_available_date = max_available_date - pd.Timedelta(days=365)
default_end_date = max_available_date
default_start_date = max_available_date - pd.Timedelta(days=90)

start_date = st.sidebar.date_input(
    "Start date",
    value=default_start_date,
    min_value=min_available_date,
    max_value=max_available_date,
    key="analysis_start_date",
)

end_date = st.sidebar.date_input(
    "End date",
    value=default_end_date,
    min_value=start_date,
    max_value=max_available_date,
    key="analysis_end_date",
)


# Period information

period_days = (end_date - start_date).days

st.sidebar.markdown(
    f"""
<div style="
    margin-top:0.45rem;
    padding:0.65rem 0.75rem;
    border-radius:10px;
    background:rgba(124,156,255,0.07);
    border:1px solid rgba(124,156,255,0.14);
">
    <div style="
        color:#64748b;
        font-size:0.68rem;
        text-transform:uppercase;
        letter-spacing:0.08em;
    ">
        Selected period
    </div>
    <div style="
        color:#e2e8f0;
        font-size:0.88rem;
        font-weight:700;
        margin-top:0.2rem;
    ">
        {period_days:,} days
    </div>
    <div style="
        color:#64748b;
        font-size:0.70rem;
        margin-top:0.15rem;
    ">
        {start_date} → {end_date}
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MARKET DATA
# ============================================================

st.sidebar.markdown(
    """
<div style="
    color:#94a3b8;
    font-size:0.70rem;
    font-weight:800;
    letter-spacing:0.13em;
    text-transform:uppercase;
    margin-top:1.35rem;
    margin-bottom:0.65rem;
">
    📡 Market Data
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# REFRESH MARKET DATA
# ============================================================

refresh_market_data = st.sidebar.button(
    "↻  Refresh Market Data",
    use_container_width=True,
)

if refresh_market_data:

    # Clear Streamlit cached results
    st.cache_data.clear()

    # Tell the data loaders to bypass their persistent/API cache
    st.session_state["force_market_refresh"] = True

    # Rerun the application so fresh data is loaded
    st.rerun()


st.sidebar.markdown(
    """
<div style="
    color:#64748b;
    font-size:0.68rem;
    line-height:1.45;
    margin-top:0.45rem;
">
    Market data is cached to reduce API requests.
    Refresh only when you need the latest available data.
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PORTFOLIO ASSETS
# ============================================================

st.sidebar.markdown(
    """
<div style="
    color:#94a3b8;
    font-size:0.70rem;
    font-weight:800;
    letter-spacing:0.13em;
    text-transform:uppercase;
    margin-top:1.35rem;
    margin-bottom:0.65rem;
">
    🪙 Portfolio Assets
</div>
""",
    unsafe_allow_html=True,
)

selected_coin_ids = st.sidebar.multiselect(
    "Select cryptocurrencies",
    options=top_coins["CoinGecko ID"].tolist(),
    default=top_coins["CoinGecko ID"].tolist()[:5],
    format_func=coin_name,
    help="Select the cryptocurrencies included in the portfolio analysis.",
)


# ============================================================
# WEIGHTING MODEL
# ============================================================

st.sidebar.markdown(
    """
<div style="
    color:#94a3b8;
    font-size:0.70rem;
    font-weight:800;
    letter-spacing:0.13em;
    text-transform:uppercase;
    margin-top:1.35rem;
    margin-bottom:0.65rem;
">
    ⚖️ Weighting Model
</div>
""",
    unsafe_allow_html=True,
)

weight_mode = st.sidebar.radio(
    "Portfolio weighting",
    options=[
        "Equal Weight",
        "Custom Weights",
    ],
    horizontal=False,
)


# ============================================================
# CUSTOM WEIGHTS
# ============================================================

if weight_mode == "Custom Weights":

    st.sidebar.markdown(
        """
<div style="
    margin-top:0.45rem;
    padding:0.65rem 0.75rem;
    border-radius:10px;
    background:rgba(251,191,36,0.06);
    border:1px solid rgba(251,191,36,0.14);
    color:#cbd5e1;
    font-size:0.72rem;
    line-height:1.45;
">
    Custom weights must add up to
    <strong>100%</strong>.
</div>
""",
        unsafe_allow_html=True,
    )

    custom_weights = {}

    for coin_id in selected_coin_ids:

        custom_weights[coin_id] = st.sidebar.number_input(
            coin_name(coin_id),
            min_value=0.0,
            max_value=1.0,
            value=1.0 / len(selected_coin_ids)
            if selected_coin_ids
            else 0.0,
            step=0.01,
            format="%.2f",
            key=f"weight_{coin_id}",
        )

    weight_sum = sum(custom_weights.values())

    if abs(weight_sum - 1.0) < 0.001:

        st.sidebar.markdown(
            f"""
<div style="
    margin-top:0.55rem;
    padding:0.55rem 0.7rem;
    border-radius:9px;
    background:rgba(52,211,153,0.08);
    border:1px solid rgba(52,211,153,0.16);
    color:#86efac;
    font-size:0.74rem;
    font-weight:700;
">
    ✓ Weights total: 100%
</div>
""",
            unsafe_allow_html=True,
        )

        portfolio_weights = pd.Series(custom_weights)

    else:

        st.sidebar.markdown(
            f"""
<div style="
    margin-top:0.55rem;
    padding:0.55rem 0.7rem;
    border-radius:9px;
    background:rgba(251,113,133,0.08);
    border:1px solid rgba(251,113,133,0.16);
    color:#fda4af;
    font-size:0.74rem;
    font-weight:700;
">
    ⚠ Weights total: {weight_sum:.1%}
</div>
""",
            unsafe_allow_html=True,
        )

        portfolio_weights = pd.Series(custom_weights)

else:

    portfolio_weights = pd.Series(
        1.0 / len(selected_coin_ids),
        index=selected_coin_ids,
    )


# ============================================================
# SIDEBAR FOOTER
# ============================================================

st.sidebar.markdown(
    """
<div style="
    margin-top:1.5rem;
    padding-top:1rem;
    border-top:1px solid rgba(148,163,184,0.12);
    color:#475569;
    font-size:0.68rem;
    line-height:1.55;
">
    <strong style="color:#64748b;">
        Crypto Portfolio Analytics
    </strong>
    <br>
    Historical data · Risk · Backtesting
    <br>
    Monte Carlo · Optimization · AI
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HISTORICAL DATA
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def load_historical_data(
    coin_ids,
    start_date,
    end_date,
    refresh=False,
):

    if not coin_ids:
        return pd.DataFrame()

    try:

        requested_days = max(
            1,
            (
                pd.Timestamp(end_date)
                - pd.Timestamp(start_date)
            ).days + 1,
        )

        # ----------------------------------------------------
        # DOWNLOAD HISTORICAL DATA
        # ----------------------------------------------------

        all_prices = download_top_100_historical_prices(
            days=requested_days,
            force_refresh=refresh,
        )

        # ----------------------------------------------------
        # VALIDATE DOWNLOAD
        # ----------------------------------------------------

        if all_prices is None:
            return pd.DataFrame()

        if all_prices.empty:
            return pd.DataFrame()

        # ----------------------------------------------------
        # KEEP ONLY SELECTED COINS
        # ----------------------------------------------------

        available_coin_ids = [
            coin_id
            for coin_id in coin_ids
            if coin_id in all_prices.columns
        ]

        if not available_coin_ids:
            return pd.DataFrame()

        prices_df = (
            all_prices[
                available_coin_ids
            ]
            .copy()
        )

        # ----------------------------------------------------
        # NORMALIZE DATE INDEX
        # ----------------------------------------------------

        prices_df.index = pd.to_datetime(
            prices_df.index,
            errors="coerce",
        )

        prices_df = prices_df[
            ~prices_df.index.isna()
        ]

        if getattr(
            prices_df.index,
            "tz",
            None,
        ) is not None:

            prices_df.index = (
                prices_df.index
                .tz_localize(None)
            )

        # ----------------------------------------------------
        # FILTER SELECTED PERIOD
        # ----------------------------------------------------

        start_timestamp = pd.Timestamp(
            start_date
        )

        end_timestamp = (
            pd.Timestamp(end_date)
            + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )

        prices_df = prices_df.loc[
            (
                prices_df.index
                >= start_timestamp
            )
            &
            (
                prices_df.index
                <= end_timestamp
            )
        ]

        # ----------------------------------------------------
        # CLEAN DATA
        # ----------------------------------------------------

        prices_df = (
            prices_df
            .sort_index()
            .ffill()
        )

        prices_df = (
            prices_df
            .dropna(
                axis=1,
                how="all",
            )
        )

        return prices_df

    except Exception as e:

        # Keep the application alive, but expose the
        # actual problem instead of silently hiding it.
        st.warning(
            f"Historical market data could not be loaded: {e}"
        )

        return pd.DataFrame()


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

force_market_refresh = (
    st.session_state.pop(
        "force_market_refresh",
        False,
    )
)


with st.spinner(
    "Loading historical cryptocurrency data..."
):

    prices_df = load_historical_data(

        tuple(selected_coin_ids),

        str(start_date),

        str(end_date),

        refresh=force_market_refresh,

    )


# ============================================================
# VALIDATE HISTORICAL DATA
#
# We no longer st.stop() here. Halting the script this early meant
# execution never reached st.tabs(...) further down, so the whole tab
# bar was missing whenever data failed to load. Instead we set a
# `data_available` flag; every tab checks it and shows a friendly
# "no data yet" message instead of crashing, but the tabs themselves
# always render.
# ============================================================

data_available = not prices_df.empty

if not data_available:

    st.warning(
        "Historical price data could not be loaded for the selected "
        "coins. The tabs below will still work, but portfolio, risk, "
        "backtest, Monte Carlo and optimization results will be empty "
        "until data is available. Try 'Refresh Market Data' in the "
        "sidebar, or confirm cached data exists in data/cache/ for "
        "these coins."
    )


# ============================================================
# KEEP ONLY AVAILABLE COINS
# ============================================================

available_coin_ids = [

    coin_id

    for coin_id in selected_coin_ids

    if coin_id in prices_df.columns

]


if data_available and not available_coin_ids:

    st.warning(
        "None of the selected cryptocurrencies have historical data "
        "available for the selected period."
    )

    data_available = False


prices_df = prices_df[
    available_coin_ids
]


# ============================================================
# UPDATE WEIGHTS IF SOME COINS ARE MISSING
# ============================================================

if data_available and len(available_coin_ids) != len(
    selected_coin_ids
):

    st.warning(
        "Some selected cryptocurrencies "
        "do not have available historical data "
        "for the selected period."
    )


selected_coin_ids = available_coin_ids


# Recalculate weights for the (possibly reduced) coin list.
#
# `portfolio_weights` (built earlier in the Equal Weight / Custom
# Weights sidebar section) already reflects the user's chosen
# weighting for the originally selected coins, so we just reindex it
# down to the coins that actually have historical data and
# renormalize. If no coins have data at all, `weights` is just empty
# instead of raising a ZeroDivisionError.

if available_coin_ids:

    weights_series = (
        portfolio_weights
        .reindex(available_coin_ids)
        .fillna(0.0)
    )

    if weights_series.sum() > 0:

        weights_series = (
            weights_series
            / weights_series.sum()
        )

    else:

        weights_series = pd.Series(
            1.0 / len(available_coin_ids),
            index=available_coin_ids,
        )

    weights = weights_series.values

else:

    weights = np.array([])


# ============================================================
# RETURNS
# ============================================================

if not prices_df.empty:

    returns = (
        prices_df
        .pct_change()
        .dropna(
            how="all"
        )
    )

    returns = (
        returns
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            how="all"
        )
    )

else:

    returns = pd.DataFrame(
        columns=available_coin_ids,
        dtype=float,
    )


if not returns.empty and len(weights) == returns.shape[1]:

    portfolio_returns = (
        returns
        .fillna(0)
        .dot(weights)
    )

else:

    portfolio_returns = pd.Series(
        dtype=float
    )


# ============================================================
# PORTFOLIO VALUE
# ============================================================

if not portfolio_returns.empty:

    portfolio_value = (
        1
        * (
            1
            + portfolio_returns
        )
        .cumprod()
    )

else:

    portfolio_value = pd.Series(
        dtype=float
    )


# ============================================================
# OPTIMIZATION CALLBACK
# ============================================================

def run_optimization_callback(
    returns_data,
    assets,
    method,
    target_volatility_value,
):

    try:

        try:
            optimization_result = calculate_weights(
                returns_data,
                method=method,
                target_volatility_value=target_volatility_value,
            )
        except TypeError:
            optimization_result = calculate_weights(
                returns_data,
                method=method,
                target_volatility=target_volatility_value,
            )

        optimization_array = np.asarray(
            optimization_result,
            dtype=float,
        ).reshape(-1)

        if len(optimization_array) != len(assets):
            raise ValueError(
                f"Optimization returned {len(optimization_array)} weights "
                f"for {len(assets)} assets."
            )

        if not np.all(np.isfinite(optimization_array)):
            raise ValueError("Optimization returned invalid weights.")

        weight_sum = optimization_array.sum()

        if weight_sum <= 0:
            raise ValueError("Optimization returned invalid portfolio weights.")

        optimization_array = optimization_array / weight_sum

        st.session_state["optimization_result"] = optimization_array
        st.session_state["optimization_signature"] = (
            tuple(assets),
            str(start_date),
            str(end_date),
        )
        st.session_state["optimization_method"] = method
        st.session_state["optimization_target_volatility"] = target_volatility_value
        st.session_state["optimization_error"] = None

    except Exception as e:
        st.session_state["optimization_result"] = None
        st.session_state["optimization_error"] = str(e)


# ============================================================
# DOCUMENTATION TAB CONTENT
# ============================================================

def render_documentation():

    st.header("📚 Project Documentation")

    st.write(
        "This tab documents what every section of the application does, "
        "where the data comes from, which parts are cached, which parts "
        "are calculated live, and how the code flows from input to result."
    )

    st.info(
        "The documentation below describes the current implementation in "
        "this project. It follows the actual Streamlit application and "
        "source modules used by the dashboard."
    )

    # ========================================================
    # 1. ARCHITECTURE
    # ========================================================

    with st.expander(
        "🏗️ 1. Application Architecture",
        expanded=True,
    ):

        st.markdown(
            """
### Main application

`app.py` is the main Streamlit entry point. It:

1. Starts the Streamlit page.
2. Loads the Top 100 cryptocurrency market dataset.
3. Builds the cryptocurrency selectors.
4. Loads historical prices for the selected portfolio.
5. Calculates returns and portfolio statistics.
6. Creates all application tabs.
7. Passes data into the specialized source modules.
8. Displays charts, tables, heatmaps and analytical results.

### Main source modules

| Module | Responsibility |
|---|---|
| `src/data.py` | External market-data downloading and data caching. |
| `src/risk.py` | Portfolio risk, VaR, CVaR, volatility, drawdown, Sharpe, Sortino, covariance and correlation. |
| `src/optimization.py` | Portfolio-weight optimization methods. |
| `src/backtest.py` | Historical portfolio strategy simulation. |
| `src/monte_carlo.py` | Random future-path simulation and probability/scenario calculations. |
| `src/crypto_research.py` | General cryptocurrency research data used by Crypto Analysis. |
| `src/single_crypto_analysis.py` | Detailed one-coin research, technical analysis and DeFi context. |
| `src/ai_analysis.py` | AI-generated portfolio interpretation. |

### High-level data flow

`External APIs / cached data`
→ `src/data.py`
→ `app.py`
→ `returns`
→ `risk / optimization / backtest / Monte Carlo`
→ `visualizations`
→ `AI context`

### Single Crypto Research flow

`CoinGecko`
→ `src/single_crypto_analysis.py`
→ `historical data + fundamentals + technical indicators + DeFi`
→ `research sub-tabs`
"""
        )

    # ========================================================
    # 2. STREAMLIT EXECUTION MODEL
    # ========================================================

    with st.expander(
        "⚙️ 2. How Streamlit Executes the Code"
    ):

        st.markdown(
            """
### Important Streamlit behavior

Streamlit normally reruns `app.py` from top to bottom whenever a widget
changes or a button is pressed.

The application therefore uses caching for downloaded data so that every
rerun does not create a new external API request.

The approximate execution order is:

1. Import Python libraries and project modules.
2. Configure the Streamlit page.
3. Load Top 100 cryptocurrency market data.
4. Normalize and clean the market dataset.
5. Read portfolio settings from the sidebar.
6. Load historical prices.
7. Calculate daily returns.
8. Validate portfolio weights.
9. Create the application tabs.
10. Execute the content of each tab.

### Tabs and execution

A Streamlit tab is not a separate Python process.

All tab blocks are part of the same Streamlit script. The tabs determine
which interface section is displayed to the user.

### Why caching matters

Without caching, every Streamlit rerun could trigger another request to
CoinGecko or another external API.

Repeated requests can result in:

- slower application performance
- unnecessary API traffic
- CoinGecko HTTP 429 rate-limit errors

Caching reduces these repeated requests.
"""
        )

    # ========================================================
    # 3. CACHING
    # ========================================================

    with st.expander(
        "💾 3. Caching: What Is Cached and What Is Not",
        expanded=True,
    ):

        cache_rows = [
            (
                "Top 100 market data",
                "app.py / src.data",
                "@st.cache_data",
                "1 hour",
                "Reused until TTL expires or cache is cleared.",
            ),
            (
                "Portfolio historical prices",
                "app.py / src.data",
                "@st.cache_data",
                "1 hour",
                "Depends on selected coins, dates and refresh parameters.",
            ),
            (
                "Single-coin detailed information",
                "src/single_crypto_analysis.py",
                "@st.cache_data",
                "1 hour",
                "CoinGecko detailed coin response is cached.",
            ),
            (
                "Single-coin historical data",
                "src/single_crypto_analysis.py",
                "@st.cache_data",
                "1 hour",
                "Historical market-chart data is cached.",
            ),
            (
                "Bitcoin history",
                "src/single_crypto_analysis.py",
                "@st.cache_data",
                "1 hour",
                "Bitcoin history used for correlation is cached.",
            ),
            (
                "DeFiLlama protocol list",
                "src/single_crypto_analysis.py",
                "@st.cache_data",
                "1 hour",
                "Protocol discovery data is cached.",
            ),
            (
                "DeFiLlama protocol details",
                "src/single_crypto_analysis.py",
                "@st.cache_data",
                "1 hour",
                "Each protocol result is cached separately.",
            ),
            (
                "Returns",
                "app.py",
                "Not cached",
                "Per rerun",
                "Calculated from the current historical prices.",
            ),
            (
                "Risk calculations",
                "app.py / src.risk",
                "Not cached",
                "Per rerun",
                "Recalculated from the current return data.",
            ),
            (
                "Correlation",
                "app.py / src.risk",
                "Not cached",
                "Per rerun",
                "Calculated from the current asset returns.",
            ),
            (
                "Covariance",
                "app.py / src.risk",
                "Not cached",
                "Per rerun",
                "Calculated from the current asset returns.",
            ),
            (
                "Optimization",
                "app.py / src.optimization",
                "Not cached",
                "When requested",
                "Calculated when the user presses Calculate Optimal Weights.",
            ),
            (
                "Backtest",
                "app.py / src.backtest",
                "Not cached",
                "When executed",
                "Calculated from the current portfolio data and strategy.",
            ),
            (
                "Monte Carlo",
                "app.py / src.monte_carlo",
                "Not cached",
                "When executed",
                "Simulation paths are generated from current inputs.",
            ),
            (
                "AI analysis",
                "app.py / src.ai_analysis",
                "Not cached",
                "When requested",
                "Generated when the user presses Generate AI Analysis.",
            ),
        ]

        st.dataframe(
            pd.DataFrame(
                cache_rows,
                columns=[
                    "Component",
                    "Code location",
                    "Caching",
                    "TTL",
                    "Behavior",
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
### `@st.cache_data`

`st.cache_data` stores the result of a function for the same input
arguments.

It is primarily used for external data because downloaded data can be
reused between Streamlit reruns.

### Refresh behavior

The application provides refresh mechanisms for market and research data.

When the relevant cache is cleared, the next request downloads fresh data.

### What is not cached

The following are intentionally recalculated:

- daily returns
- portfolio returns
- portfolio value
- risk metrics
- correlation
- covariance
- portfolio variance
- portfolio volatility
- optimization results
- backtest results
- Monte Carlo simulations
- displayed charts

These calculations depend on current inputs and are therefore not stored
as independent cached results.
"""
        )

    # ========================================================
    # 4. TAB-BY-TAB DOCUMENTATION
    # ========================================================

    with st.expander(
        "📑 4. Tab-by-Tab Documentation",
        expanded=True,
    ):

        tab_docs = {

            "📊 Portfolio": {
                "purpose": (
                    "Build a cryptocurrency portfolio, assign weights "
                    "and evaluate historical portfolio behavior."
                ),
                "inputs": (
                    "Selected cryptocurrencies, portfolio weight method, "
                    "custom weights and selected start/end dates."
                ),
                "data": (
                    "Top 100 cryptocurrency market data and historical "
                    "price data."
                ),
                "workflow": (
                    "Select cryptocurrencies → determine weights → load "
                    "historical prices → calculate returns → calculate "
                    "portfolio returns → display portfolio performance."
                ),
                "calculations": (
                    "Daily returns, weighted portfolio returns, cumulative "
                    "portfolio value and portfolio performance statistics."
                ),
                "caching": (
                    "Market data and historical price downloads are cached."
                ),
                "not_cached": (
                    "Portfolio calculations and charts are recalculated "
                    "from the loaded DataFrames."
                ),
            },

            "🔎 Crypto Analysis": {
                "purpose": (
                    "Inspect one cryptocurrency using market and research "
                    "information."
                ),
                "inputs": (
                    "One cryptocurrency selected from the available list."
                ),
                "data": (
                    "Current market data, cryptocurrency research data and "
                    "historical price data."
                ),
                "workflow": (
                    "Select coin → load market/research information → "
                    "display fundamentals and historical performance."
                ),
                "calculations": (
                    "Descriptive market statistics and historical price "
                    "visualization."
                ),
                "caching": (
                    "Underlying data functions may use Streamlit caching."
                ),
                "not_cached": (
                    "Displayed metrics and charts are rendered again on rerun."
                ),
            },

            "🔬 Single Crypto Research": {
                "purpose": (
                    "Perform detailed research for one cryptocurrency "
                    "independently from the main portfolio."
                ),
                "inputs": (
                    "Selected cryptocurrency, refresh action and analysis period."
                ),
                "data": (
                    "CoinGecko detailed coin data, historical market-chart "
                    "data, Bitcoin history and DeFiLlama data."
                ),
                "workflow": (
                    "Select coin → load detailed data → detect available "
                    "historical range → select analysis period → calculate "
                    "indicators → display research sub-tabs."
                ),
                "calculations": (
                    "Returns, annualized volatility, RSI(14), trend, drawdown, "
                    "BTC correlation, moving averages, market-cap/FDV and "
                    "supply ratios."
                ),
                "caching": (
                    "External research requests are cached for one hour."
                ),
                "not_cached": (
                    "Technical indicators, tables and charts are recalculated "
                    "from the loaded historical data."
                ),
            },

            "⚠️ Risk & Performance": {
                "purpose": (
                    "Measure portfolio risk, risk-adjusted performance and "
                    "relationships between the selected cryptocurrencies."
                ),
                "inputs": (
                    "Portfolio returns, portfolio weights, selected dates "
                    "and optimized weights when available."
                ),
                "data": (
                    "The historical price dataset and corresponding daily "
                    "return DataFrame."
                ),
                "workflow": (
                    "Load returns → determine active portfolio weights → "
                    "calculate risk metrics → calculate correlation and "
                    "covariance → display metrics and heatmaps."
                ),
                "calculations": (
                    "Annualized return, annualized volatility, downside "
                    "deviation, Sharpe ratio, Sortino ratio, historical VaR, "
                    "CVaR, maximum drawdown, correlation, covariance, "
                    "portfolio variance and portfolio volatility."
                ),
                "caching": (
                    "No dedicated Streamlit cache is used for risk calculations."
                ),
                "not_cached": (
                    "Risk metrics, correlation, covariance and portfolio "
                    "statistics are recalculated from the current returns."
                ),
            },

            "📈 Backtest": {
                "purpose": (
                    "Test historical portfolio strategies using historical data."
                ),
                "inputs": (
                    "Selected strategy, portfolio, historical period and "
                    "rebalance settings where applicable."
                ),
                "data": (
                    "Historical portfolio prices and returns."
                ),
                "workflow": (
                    "Select strategy → simulate historical portfolio behavior "
                    "→ calculate performance metrics → display results."
                ),
                "calculations": (
                    "Strategy-specific historical performance calculations "
                    "implemented in `src/backtest.py`."
                ),
                "caching": (
                    "No dedicated Streamlit cache."
                ),
                "not_cached": (
                    "Backtest results are calculated from the current data."
                ),
            },

            "🎲 Monte Carlo": {
                "purpose": (
                    "Generate simulated future portfolio paths and scenario "
                    "statistics."
                ),
                "inputs": (
                    "Simulation count, forecast horizon and current portfolio "
                    "return characteristics."
                ),
                "data": (
                    "Current portfolio return series."
                ),
                "workflow": (
                    "Take historical return characteristics → generate random "
                    "future paths → summarize simulated outcomes."
                ),
                "calculations": (
                    "Monte Carlo paths and probability/scenario statistics "
                    "implemented in `src/monte_carlo.py`."
                ),
                "caching": (
                    "No dedicated Streamlit cache."
                ),
                "not_cached": (
                    "Simulation output is regenerated when the simulation runs."
                ),
            },

            "⚙️ Optimization": {
                "purpose": (
                    "Calculate portfolio weights using different portfolio "
                    "construction methods."
                ),
                "inputs": (
                    "Optimization method and target volatility when required."
                ),
                "data": (
                    "Current asset return DataFrame."
                ),
                "workflow": (
                    "Take returns → select optimization method → calculate "
                    "weights → validate and normalize weights → store the "
                    "result → display the optimized allocation."
                ),
                "calculations": (
                    "Equal Weight, Inverse Volatility, Minimum Volatility, "
                    "Maximum Sharpe, Risk Parity, Maximum Diversification "
                    "and Target Volatility."
                ),
                "caching": (
                    "Optimization results are not stored using `st.cache_data`."
                ),
                "not_cached": (
                    "Optimization is explicitly recalculated when the user "
                    "presses Calculate Optimal Weights."
                ),
            },

            "🗃️ Data": {
                "purpose": (
                    "Display the datasets currently used by the application."
                ),
                "inputs": (
                    "Current application data."
                ),
                "data": (
                    "Top 100 cryptocurrency market data and historical "
                    "price observations."
                ),
                "workflow": (
                    "Load normalized data → select relevant columns → "
                    "display DataFrames."
                ),
                "calculations": (
                    "Mostly formatting and data selection."
                ),
                "caching": (
                    "Data comes from the existing cached/loaded data functions."
                ),
                "not_cached": (
                    "The tables themselves are rendered from current "
                    "in-memory DataFrames."
                ),
            },

            "🤖 AI Analysis": {
                "purpose": (
                    "Generate an AI-based interpretation of the current "
                    "portfolio context."
                ),
                "inputs": (
                    "Selected coins, active weights, dates, return, volatility, "
                    "Sharpe, Sortino, VaR, CVaR, drawdown and available "
                    "simulation information."
                ),
                "data": (
                    "Portfolio metrics calculated directly in `app.py`."
                ),
                "workflow": (
                    "Press Generate AI Analysis → build portfolio context → "
                    "send context to `generate_ai_analysis()` → display result."
                ),
                "calculations": (
                    "The application supplies numerical portfolio context. "
                    "The AI module generates the textual interpretation."
                ),
                "caching": (
                    "The AI button action is not cached with `st.cache_data`."
                ),
                "not_cached": (
                    "AI generation is triggered by the button and is not "
                    "stored as a cached API response."
                ),
            },
        }

        for title, doc in tab_docs.items():

            st.markdown(f"## {title}")

            st.markdown(
                f"**What it does:** {doc['purpose']}"
            )

            st.markdown(
                f"**Inputs:** {doc['inputs']}"
            )

            st.markdown(
                f"**Data source:** {doc['data']}"
            )

            st.markdown(
                f"**How it works:** {doc['workflow']}"
            )

            st.markdown(
                f"**Calculations / code:** {doc['calculations']}"
            )

            st.markdown(
                f"**Caching:** {doc['caching']}"
            )

            st.markdown(
                f"**Not cached / recalculated:** {doc['not_cached']}"
            )

            st.divider()

    # ========================================================
    # 5. SINGLE CRYPTO SUB-TABS
    # ========================================================

    with st.expander(
        "🔬 5. Single Crypto Research Sub-Tab Documentation"
    ):

        subtab_docs = [

            (
                "📊 Overview",
                "Market snapshot and broad performance picture. "
                "Uses the selected period's historical price Series "
                "and CoinGecko market data."
            ),

            (
                "📈 Technical",
                "Historical trend and technical indicators including "
                "RSI(14), returns, drawdown and 20D/50D/200D moving "
                "averages when enough observations exist."
            ),

            (
                "🪙 Tokenomics",
                "Circulating, total and maximum supply plus "
                "circulating/max-supply and market-cap/FDV ratios. "
                "Unavailable vesting information is not invented."
            ),

            (
                "🏗️ Project",
                "Project description, CoinGecko categories, homepage, "
                "blockchain explorer and GitHub links when supplied "
                "by the API."
            ),

            (
                "🌐 DeFi",
                "Attempts to identify a related DeFiLlama protocol and "
                "display available protocol-level information such as "
                "TVL when available."
            ),

            (
                "📝 Research Summary",
                "Combines market, technical, tokenomics and project "
                "observations into a research-oriented summary."
            ),
        ]

        for name, description in subtab_docs:

            st.markdown(
                f"**{name}** — {description}"
            )

    # ========================================================
    # 6. CORE CALCULATIONS
    # ========================================================

    with st.expander(
        "🧮 6. Core Calculations and Formulas"
    ):

        st.markdown(
            """
### Daily return

`r_t = P_t / P_(t-1) - 1`

The application calculates daily percentage returns from historical prices.

### Portfolio return

`R_p,t = Σ(w_i × r_i,t)`

Each asset return is multiplied by its portfolio weight and the results
are summed.

### Annualized return

The application annualizes daily portfolio returns using the project's
365-day crypto-market convention.

### Annualized volatility

`σ_annual = σ_daily × √365`

The project uses 365 days because cryptocurrency markets operate
continuously throughout the year.

### Maximum drawdown

`Drawdown_t = Wealth_t / RunningPeak_t - 1`

Maximum drawdown is the minimum value of the drawdown series.

### Sharpe ratio

The Sharpe ratio compares portfolio excess return with portfolio
volatility using the configured risk-free rate.

### Sortino ratio

The Sortino ratio evaluates return relative to downside deviation,
focusing on negative returns rather than total volatility.

### Historical VaR

Historical Value at Risk uses the lower-tail percentile of the
portfolio return distribution.

For example, 95% VaR uses the 5th percentile of returns.

### CVaR

Conditional Value at Risk calculates the average return of observations
that are at or below the selected VaR threshold.

### Correlation matrix

The correlation matrix measures the linear relationship between the
returns of every pair of selected cryptocurrencies.

Values range from:

- `+1` — strong positive relationship
- `0` — little or no linear relationship
- `-1` — strong negative relationship

The Risk & Performance tab displays this matrix as a heatmap.

The diagonal normally contains `1.00` because every asset is perfectly
correlated with itself.

### Covariance matrix

The covariance matrix measures how two cryptocurrency returns move
together in absolute return units.

Positive covariance means the assets tend to move in the same direction.

Negative covariance means the assets tend to move in opposite directions.

The Risk & Performance tab displays the covariance matrix as a heatmap.

### Portfolio variance

`σ²_p = wᵀΣw`

where:

- `w` = portfolio weight vector
- `Σ` = annualized covariance matrix

Portfolio variance combines the individual asset variances and the
pairwise covariance relationships between assets.

### Portfolio volatility

`σ_p = √(wᵀΣw)`

Portfolio volatility is the square root of portfolio variance.

When optimized weights are available and valid, the Risk & Performance
calculations use those optimized weights.

### BTC correlation

Single Crypto Research aligns the selected cryptocurrency's daily
returns with Bitcoin daily returns and calculates Pearson correlation
after removing missing pairs.
"""
        )

    # ========================================================
    # 7. CORRELATION / COVARIANCE HEATMAP EXPLANATION
    # ========================================================

    with st.expander(
        "🔥 7. Understanding the Risk Heatmaps"
    ):

        st.markdown(
            """
### Correlation Heatmap

The correlation heatmap is a visual representation of the correlation
matrix.

Each cell represents the correlation between two cryptocurrencies.

#### How to read it

- Values close to `+1` indicate similar return movements.
- Values close to `0` indicate weak linear relationships.
- Values close to `-1` indicate opposite return movements.

The diagonal is `1.00` because each cryptocurrency is compared with itself.

The heatmap is useful for identifying diversification relationships
between portfolio assets.

### Covariance Heatmap

The covariance heatmap represents the annualized covariance matrix.

Unlike correlation, covariance is not restricted to the range `-1` to `+1`.

Its magnitude depends on the volatility of the assets.

The covariance matrix is particularly important because it is used in
portfolio variance:

`σ²_p = wᵀΣw`

Therefore, the covariance relationships between assets directly affect
the overall portfolio risk.

### Important difference

Correlation answers:

> How similarly do two assets move?

Covariance answers:

> How much do two assets move together in return units?

Correlation is standardized, while covariance depends on the scale and
volatility of the assets.
"""
        )

    # ========================================================
    # 8. DATA AVAILABILITY AND API LIMITATIONS
    # ========================================================

    with st.expander(
        "⚠️ 8. Data Availability, API Limits and Important Limitations"
    ):

        st.markdown(
            """
### CoinGecko rate limits

The application uses CoinGecko for market and detailed cryptocurrency
information.

Repeated requests can result in:

`HTTP 429 — Too Many Requests`

Caching is therefore an important part of the application architecture.

### Portfolio historical range

The portfolio calculations depend on the historical price data returned
by the portfolio data-loading function.

The available historical period therefore depends on the data returned
by the external provider.

### Single Crypto historical range

Single Crypto Research requests historical data for the selected coin and
detects the actual available historical range.

The analysis period must therefore fall within the available historical
data.

If a selected cryptocurrency does not have enough historical data for the
requested period, the application displays an appropriate message rather
than creating artificial historical observations.

### Token unlocks

The current Single Crypto Research implementation does not attempt to
invent detailed vesting schedules or unlock percentages when reliable
data is unavailable.

### AI analysis

The AI tab receives structured numerical portfolio information from the
application.

The underlying portfolio calculations remain the source of the numerical
metrics. The AI output is an interpretation of those calculated inputs.
"""
        )

    # ========================================================
    # 9. CODE MAP
    # ========================================================

    with st.expander(
        "🧩 9. Code Map — Where to Change Things"
    ):

        code_map = pd.DataFrame(
            [
                (
                    "Top 100 API / market data",
                    "src/data.py",
                    "download_top_coins()"
                ),
                (
                    "Historical market data",
                    "src/data.py",
                    "Historical download functions"
                ),
                (
                    "Portfolio historical loader",
                    "app.py",
                    "load_historical_data()"
                ),
                (
                    "Daily returns",
                    "app.py / src.risk",
                    "Return calculation logic"
                ),
                (
                    "Risk calculations",
                    "src/risk.py",
                    "Portfolio variance, VaR, CVaR, Sharpe, Sortino, drawdown, covariance and correlation"
                ),
                (
                    "Correlation heatmap",
                    "app.py",
                    "Risk & Performance tab"
                ),
                (
                    "Covariance heatmap",
                    "app.py",
                    "Risk & Performance tab"
                ),
                (
                    "Portfolio variance",
                    "app.py / src.risk",
                    "wᵀΣw"
                ),
                (
                    "Optimization",
                    "src/optimization.py",
                    "calculate_weights() and optimization methods"
                ),
                (
                    "Optimization state",
                    "app.py",
                    "st.session_state['optimization_result']"
                ),
                (
                    "Backtesting",
                    "src/backtest.py",
                    "Strategy and backtest calculations"
                ),
                (
                    "Monte Carlo",
                    "src/monte_carlo.py",
                    "Simulation functions"
                ),
                (
                    "General crypto research",
                    "src/crypto_research.py",
                    "get_coin_research()"
                ),
                (
                    "Single crypto research",
                    "src/single_crypto_analysis.py",
                    "render_single_crypto_research()"
                ),
                (
                    "DeFi data",
                    "src/single_crypto_analysis.py",
                    "DeFiLlama functions"
                ),
                (
                    "AI analysis",
                    "src/ai_analysis.py",
                    "generate_ai_analysis()"
                ),
                (
                    "UI / tabs",
                    "app.py",
                    "st.tabs() and tab blocks"
                ),
            ],
            columns=[
                "Feature",
                "File",
                "Main function / location",
            ],
        )

        st.dataframe(
            code_map,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # 10. PRACTICAL DEBUGGING
    # ========================================================

    with st.expander(
        "🛠️ 10. Practical Debugging Guide"
    ):

        st.markdown(
            """
### If Top 100 data fails

Check `src/data.py` and the CoinGecko request/cache.

The main function to inspect is the Top 100 market-data downloader.

### If portfolio historical data fails

Check the portfolio historical-data loader in `app.py` and the underlying
historical-data functions in `src/data.py`.

### If Single Crypto Research shows HTTP 429

Avoid repeatedly pressing refresh.

The detailed CoinGecko requests are cached and should normally be reused
until the cache expires or is explicitly cleared.

### If a technical indicator is N/A

Check the number of available historical observations.

Indicators such as 20D, 50D and 200D moving averages require enough
historical observations.

### If the correlation heatmap looks unusual

Check:

1. The selected cryptocurrencies.
2. The selected date range.
3. The number of overlapping return observations.
4. Missing values in the return DataFrame.

### If the covariance heatmap looks unusual

Remember that covariance depends on asset volatility and is not
standardized like correlation.

Also verify the annualization convention used by `src/risk.py`.

### If portfolio volatility looks different after optimization

Check whether an optimization result exists in
`st.session_state`.

When valid optimized weights exist, Risk & Performance uses those weights
instead of the original sidebar portfolio weights.

### If optimization produces unexpected weights

Inspect:

`src/optimization.py`

and verify:

- the return DataFrame is valid
- enough observations are available
- all selected assets have usable returns
- the optimization method is appropriate
- the resulting weights are finite
- the weights are normalized

### If optimization becomes stale

The application uses an optimization signature based on the selected
portfolio and analysis dates.

When the relevant portfolio configuration changes, the previous
optimization result should no longer be used.

### If Backtest or Monte Carlo results look stale

Check the Streamlit session state and confirm that the portfolio/date
signature has changed when the underlying portfolio has changed.

### If AI analysis fails

Inspect `src/ai_analysis.py` and verify that the portfolio context contains
valid numerical values before the AI function is called.
"""
        )

# ============================================================
# TABS
# ============================================================

(
    tab_portfolio,
    # tab_analysis,
    tab_single_crypto,
    tab_risk,
    tab_backtest,
    tab_monte_carlo,
    tab_optimization,
    tab_data,
    tab_ai,
    tab_documentation,
) = st.tabs(
    [
        "📊 Portfolio",
        # "🔎 Crypto Analysis",
        "🔬 Single Crypto Research",
        "⚠️ Risk & Performance",
        "📈 Backtest",
        "🎲 Monte Carlo",
        "⚙️ Optimization",
        "🗃️ Data",
        "🤖 AI Analysis",
        "📚 Documentation",
    ]
)


# ============================================================
# ACTIVE PORTFOLIO WEIGHTS
# ============================================================

portfolio_weights = np.asarray(
    weights,
    dtype=float,
).reshape(-1)

using_optimized_portfolio_weights = False

stored_signature = st.session_state.get(
    "optimization_signature"
)

current_signature = (
    tuple(available_coin_ids),
    str(start_date),
    str(end_date),
)

optimization_result = st.session_state.get(
    "optimization_result"
)


if (
    optimization_result is not None
    and stored_signature == current_signature
):

    try:

        optimized_array = np.asarray(
            optimization_result,
            dtype=float,
        ).reshape(-1)

        if (
            len(optimized_array)
            == len(available_coin_ids)

            and np.all(
                np.isfinite(
                    optimized_array
                )
            )

            and optimized_array.sum() > 0
        ):

            portfolio_weights = (
                optimized_array
                / optimized_array.sum()
            )

            using_optimized_portfolio_weights = True

    except Exception:

        using_optimized_portfolio_weights = False


portfolio_weights = pd.Series(
    portfolio_weights,
    index=available_coin_ids,
    dtype=float,
)

portfolio_weights = (
    portfolio_weights
    .reindex(available_coin_ids)
    .fillna(0.0)
)


if portfolio_weights.sum() <= 0:

    # No valid weights — typically because no historical data is
    # available yet. Don't halt the script; leave portfolio_weights
    # empty and let each tab show its own "no data" message.
    portfolio_weights = pd.Series(dtype=float)

else:

    portfolio_weights = (
        portfolio_weights
        / portfolio_weights.sum()
    )

# ============================================================
# PORTFOLIO TAB
# ============================================================

with tab_portfolio:

    st.header(
        "Cryptocurrency Portfolio"
    )

    
    # --------------------------------------------------------
    # ACTIVE WEIGHT SOURCE
    # --------------------------------------------------------

    if using_optimized_portfolio_weights:

        st.success(
            "Portfolio is using optimized weights "
            f"({st.session_state.get('optimization_method', 'Optimization')})."
        )

    else:

        st.info(
            "Portfolio is using the weights selected in the sidebar."
        )


    # --------------------------------------------------------
    # PORTFOLIO RETURNS USING ACTIVE WEIGHTS
    # --------------------------------------------------------

    if (
        not returns.empty
        and not portfolio_weights.empty
        and len(portfolio_weights) == len(returns.columns)
    ):

        active_portfolio_returns = (
            returns
            .fillna(0)
            .dot(portfolio_weights)
        )

    else:

        active_portfolio_returns = pd.Series(
            dtype=float
        )


    # --------------------------------------------------------
    # PORTFOLIO VALUE
    # --------------------------------------------------------

    if not active_portfolio_returns.empty:

        active_portfolio_value = (
            1
            + active_portfolio_returns
        ).cumprod()

    else:

        active_portfolio_value = pd.Series(
            dtype=float
        )


    # --------------------------------------------------------
    # PORTFOLIO METRICS
    # --------------------------------------------------------

    if not active_portfolio_value.empty:

        total_return = (
            active_portfolio_value.iloc[-1]
            - 1
        )

        portfolio_volatility = (
            active_portfolio_returns.std()
            * np.sqrt(365)
        )

        annual_return = (
            active_portfolio_returns.mean()
            * 365
        )

    else:

        total_return = np.nan
        portfolio_volatility = np.nan
        annual_return = np.nan

        st.warning(
            "Portfolio performance cannot be calculated because "
            "there is no historical return data for the selected "
            "cryptocurrencies and analysis period."
        )


    # --------------------------------------------------------
    # ACTIVE WEIGHTS TABLE
    # --------------------------------------------------------

    st.subheader(
        "Portfolio Allocation"
    )


    allocation_df = pd.DataFrame(

        {
            "Cryptocurrency": [
                coin_name(coin_id)
                for coin_id in available_coin_ids
            ],

            "CoinGecko ID": available_coin_ids,

            "Weight": portfolio_weights.values,

        }

    )


    st.dataframe(

        allocation_df.style.format(
            {
                "Weight": "{:.2%}",
            }
        ),

        use_container_width=True,

        hide_index=True,

    )


    # --------------------------------------------------------
    # ALLOCATION CHART
    # --------------------------------------------------------

    fig_allocation = go.Figure(

        data=[

            go.Pie(

                labels=allocation_df[
                    "Cryptocurrency"
                ],

                values=allocation_df[
                    "Weight"
                ],

                hole=0.4,

            )

        ]

    )


    fig_allocation.update_layout(

        title=(
            "Optimized Portfolio Allocation"
            if using_optimized_portfolio_weights
            else "Portfolio Allocation"
        ),

        height=450,

    )


    st.plotly_chart(

        fig_allocation,

        use_container_width=True,

    )


    # --------------------------------------------------------
    # PORTFOLIO GROWTH
    # --------------------------------------------------------

    st.subheader(
        "Portfolio Growth"
    )


    portfolio_chart = go.Figure()


    portfolio_chart.add_trace(

        go.Scatter(

            x=active_portfolio_value.index,

            y=active_portfolio_value.values,

            mode="lines",

            name=(
                "Optimized Portfolio"
                if using_optimized_portfolio_weights
                else "Portfolio"
            ),

        )

    )


    portfolio_chart.update_layout(

        xaxis_title="Date",

        yaxis_title="Portfolio Value",

        height=500,

    )


    st.plotly_chart(

        portfolio_chart,

        use_container_width=True,

    )


    # --------------------------------------------------------
    # INDIVIDUAL CRYPTOCURRENCY PRICES
    # --------------------------------------------------------

    st.subheader(
        "Individual Cryptocurrency Prices"
    )


    price_chart = go.Figure()


    for coin_id in prices_df.columns:

        price_chart.add_trace(

            go.Scatter(

                x=prices_df.index,

                y=prices_df[coin_id],

                mode="lines",

                name=coin_name(coin_id),

            )

        )


    price_chart.update_layout(

        xaxis_title="Date",

        yaxis_title="Price (USD)",

        height=500,

    )


    st.plotly_chart(

        price_chart,

        use_container_width=True,

    )


# ============================================================
# SINGLE CRYPTO RESEARCH TAB
# ============================================================

with tab_single_crypto:

    st.header("🔬 Single Cryptocurrency Research")

    st.write(
        "Analyze one cryptocurrency using market data, historical price "
        "tendencies, technical indicators, tokenomics, project information "
        "and DeFi context."
    )

    single_crypto_options = {}

    for _, row in top_coins.iterrows():

        coin_id = row.get("CoinGecko ID")
        symbol = row.get("Symbol", "")
        name = row.get("Name", "")

        if coin_id:
            rank = row.get("Rank")

            if pd.notna(rank):
                label = f"#{int(rank)} — {name} ({symbol})"
            else:
                label = f"{name} ({symbol})"

            single_crypto_options[label] = {
                "id": coin_id,
                "name": name,
                "symbol": symbol,
            }

    selected_single_label = st.selectbox(
        "Select cryptocurrency",
        options=list(single_crypto_options.keys()),
        key="single_crypto_research_selector",
    )

    selected_single_coin = single_crypto_options[selected_single_label]

    refresh_single_research = st.button(
        "🔄 Refresh Single Crypto Research",
        key="refresh_single_crypto_research",
    )

    render_single_crypto_research(
        coin_id=selected_single_coin["id"],
        coin_name=selected_single_coin["name"],
        coin_symbol=selected_single_coin["symbol"],
        top_coins=top_coins,
        portfolio_prices_df=prices_df,
        refresh=refresh_single_research,
    )


# ============================================================
# RISK & PERFORMANCE TAB
# ============================================================

with tab_risk:

    st.header("Risk & Performance Analysis")

    # Use optimized weights when they belong to the current portfolio/date range.
    risk_weights = np.asarray(weights, dtype=float).reshape(-1)
    using_optimized_weights = False

    stored_signature = st.session_state.get("optimization_signature")
    current_signature = (
        tuple(available_coin_ids),
        str(start_date),
        str(end_date),
    )

    optimization_result = st.session_state.get("optimization_result")

    if optimization_result is not None and stored_signature == current_signature:
        try:
            optimized_array = np.asarray(optimization_result, dtype=float).reshape(-1)
            if (
                len(optimized_array) == len(returns.columns)
                and np.all(np.isfinite(optimized_array))
                and optimized_array.sum() > 0
            ):
                risk_weights = optimized_array / optimized_array.sum()
                using_optimized_weights = True
        except Exception:
            using_optimized_weights = False

    risk_weights = pd.Series(
        risk_weights,
        index=returns.columns,
        dtype=float,
    ).reindex(returns.columns).fillna(0.0)

    if risk_weights.sum() <= 0:
        st.error("Invalid portfolio weights for risk analysis.")
        st.stop()

    risk_weights = risk_weights / risk_weights.sum()

    if using_optimized_weights:
        st.success(
            "Risk & Performance is using the optimized portfolio weights "
            f"({st.session_state.get('optimization_method', 'Optimization')})."
        )
    else:
        st.info("Risk & Performance is using the sidebar portfolio weights.")

    st.subheader("Weights Used for Risk Analysis")

    risk_weights_df = pd.DataFrame({
        "Cryptocurrency": [coin_name(c) for c in returns.columns],
        "CoinGecko ID": list(returns.columns),
        "Weight": risk_weights.values,
    })

    st.dataframe(
        risk_weights_df.style.format({"Weight": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

    # Portfolio returns using the selected risk weights.
    risk_portfolio_returns = returns.fillna(0).dot(risk_weights)
    risk_portfolio_value = (1 + risk_portfolio_returns).cumprod()

    annual_return = risk_portfolio_returns.mean() * 365
    annual_volatility = risk_portfolio_returns.std() * np.sqrt(365)

    sharpe = (
        annual_return / annual_volatility
        if annual_volatility != 0
        else np.nan
    )

    negative_returns = risk_portfolio_returns[
        risk_portfolio_returns < 0
    ]

    downside_deviation = (
        negative_returns.std() * np.sqrt(365)
        if len(negative_returns) > 1
        else np.nan
    )

    sortino = (
        annual_return / downside_deviation
        if pd.notna(downside_deviation) and downside_deviation != 0
        else np.nan
    )

    running_max = risk_portfolio_value.cummax()
    drawdown = risk_portfolio_value / running_max - 1
    max_drawdown = drawdown.min()

    var_95 = risk_portfolio_returns.quantile(0.05)

    cvar_95 = risk_portfolio_returns[
        risk_portfolio_returns <= var_95
    ].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Annual Return", f"{annual_return:.2%}")
        st.metric("Annual Volatility", f"{annual_volatility:.2%}")

    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Sortino Ratio", f"{sortino:.2f}")

    with col3:
        st.metric("VaR 95%", f"{var_95:.2%}")
        st.metric("CVaR 95%", f"{cvar_95:.2%}")

    st.subheader("Drawdown")

    drawdown_chart = go.Figure()
    drawdown_chart.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="Drawdown",
        )
    )
    drawdown_chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown",
        height=400,
    )
    st.plotly_chart(drawdown_chart, use_container_width=True)

    st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")

    st.subheader("Individual Cryptocurrency Risk")

    individual_risk = []

    for coin_id in returns.columns:
        coin_returns = returns[coin_id].dropna()
        volatility = coin_returns.std() * np.sqrt(365)
        var = coin_returns.quantile(0.05)

        individual_risk.append({
            "Cryptocurrency": coin_name(coin_id),
            "Annual Volatility": volatility,
            "VaR 95%": var,
        })

    risk_df = pd.DataFrame(individual_risk)

    st.dataframe(
        risk_df.style.format({
            "Annual Volatility": "{:.2%}",
            "VaR 95%": "{:.2%}",
        }),
        use_container_width=True,
    )
    # ============================================================
    # COVARIANCE HEATMAP
    # ============================================================

    st.subheader("Cryptocurrency Covariance Matrix")

    cov_matrix = covariance_matrix(returns)

    cov_display = cov_matrix.copy()

    cov_display.index = [
        coin_name(coin_id)
        for coin_id in cov_display.index
    ]

    cov_display.columns = [
        coin_name(coin_id)
        for coin_id in cov_display.columns
    ]

    fig_cov = go.Figure(
        data=go.Heatmap(
            z=cov_display.values,
            x=cov_display.columns,
            y=cov_display.index,
            colorscale="Viridis",
            text=np.round(cov_display.values, 6),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(
                title="Covariance"
            ),
            hovertemplate=(
                "%{y} vs %{x}"
                "<br>Covariance: %{z:.6f}"
                "<extra></extra>"
            ),
        )
    )

    fig_cov.update_layout(
        height=max(450, len(cov_display) * 45),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    st.plotly_chart(
        fig_cov,
        use_container_width=True,
    )

    # ============================================================
    # CORRELATION HEATMAP
    # ============================================================

    st.subheader("Cryptocurrency Correlation Matrix")

    corr_matrix = correlation_matrix(returns)

    corr_display = corr_matrix.copy()

    corr_display.index = [
        coin_name(coin_id)
        for coin_id in corr_display.index
    ]

    corr_display.columns = [
        coin_name(coin_id)
        for coin_id in corr_display.columns
    ]

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr_display.values,
            x=corr_display.columns,
            y=corr_display.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=np.round(corr_display.values, 2),
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(
                title="Correlation"
            ),
            hovertemplate=(
                "%{y} vs %{x}"
                "<br>Correlation: %{z:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig_corr.update_layout(
        height=max(450, len(corr_display) * 45),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True,
    )


# ============================================================
# BACKTEST TAB
# ============================================================

with tab_backtest:

    st.header(
        "Portfolio Backtesting"
    )


    strategy = st.selectbox(

        "Select strategy",

        [

            "Buy & Hold",

            "Periodic Rebalancing",

        ],

    )


    if strategy == "Buy & Hold":

        backtest_result = buy_and_hold(

            prices_df,

            weights,

        )

    else:

        rebalance_frequency = st.selectbox(

            "Rebalancing frequency",

            [

                "Daily",

                "Weekly",

                "Monthly",

                "Quarterly",

            ],

        )


        frequency_map = {

            "Daily": "D",

            "Weekly": "W",

            "Monthly": "M",

            "Quarterly": "Q",

        }


        backtest_result = periodic_rebalance(

            prices_df,

            weights,

            frequency=frequency_map[
                rebalance_frequency
            ],

        )


    if isinstance(
        backtest_result,
        pd.Series,
    ):

        backtest_value = backtest_result

    elif isinstance(
        backtest_result,
        pd.DataFrame,
    ):

        if "Portfolio" in backtest_result.columns:

            backtest_value = (
                backtest_result[
                    "Portfolio"
                ]
            )

        else:

            backtest_value = (
                backtest_result.iloc[:, 0]
            )

    else:

        backtest_value = pd.Series(
            backtest_result
        )


    st.subheader(
        "Backtest Performance"
    )


    backtest_chart = go.Figure()


    backtest_chart.add_trace(

        go.Scatter(

            x=backtest_value.index,

            y=backtest_value.values,

            mode="lines",

            name=strategy,

        )

    )


    backtest_chart.update_layout(

        xaxis_title="Date",

        yaxis_title="Portfolio Value",

        height=500,

    )


    st.plotly_chart(

        backtest_chart,

        use_container_width=True,

    )


    try:

        metrics = backtest_metrics(
            backtest_value
        )


        if isinstance(
            metrics,
            dict,
        ):

            st.subheader(
                "Backtest Metrics"
            )


            metric_cols = st.columns(
                len(metrics)
            )


            for column, (
                metric_name,
                metric_value,
            ) in zip(
                metric_cols,
                metrics.items(),
            ):

                with column:

                    if isinstance(
                        metric_value,
                        (int, float, np.number),
                    ):

                        st.metric(

                            metric_name,

                            (
                                f"{metric_value:.2%}"
                                if abs(metric_value)
                                <= 1

                                else
                                f"{metric_value:.2f}"
                            ),

                        )

                    else:

                        st.metric(

                            metric_name,

                            str(metric_value),

                        )


    except Exception:

        pass


# ============================================================
# MONTE CARLO TAB
# ============================================================

with tab_monte_carlo:

    st.header(
        "Monte Carlo Simulation"
    )


    col1, col2 = st.columns(2)


    with col1:

        simulations = st.number_input(

            "Number of simulations",

            min_value=100,

            max_value=10000,

            value=1000,

            step=100,

        )


    with col2:

        horizon = st.number_input(

            "Forecast horizon (days)",

            min_value=30,

            max_value=3650,

            value=252,

            step=30,

        )


    if st.button(
        "Run Monte Carlo Simulation"
    ):

        try:

            simulation_result = (
                monte_carlo_simulation(

                    portfolio_returns,

                    simulations=int(
                        simulations
                    ),

                    days=int(
                        horizon
                    ),

                )
            )


            st.session_state[
                "monte_carlo_result"
            ] = simulation_result


        except TypeError:

            try:

                simulation_result = (
                    monte_carlo_simulation(

                        portfolio_returns,

                        int(simulations),

                        int(horizon),

                    )
                )


                st.session_state[
                    "monte_carlo_result"
                ] = simulation_result


            except Exception as e:

                st.error(
                    f"Monte Carlo simulation failed: {e}"
                )


        except Exception as e:

            st.error(
                f"Monte Carlo simulation failed: {e}"
            )


    if (
        "monte_carlo_result"
        in st.session_state
    ):

        simulation_result = (
            st.session_state[
                "monte_carlo_result"
            ]
        )


        st.subheader(
            "Simulation Results"
        )


        if isinstance(
            simulation_result,
            pd.DataFrame,
        ):

            simulation_df = (
                simulation_result
            )

        else:

            simulation_df = pd.DataFrame(
                simulation_result
            )


        st.dataframe(

            simulation_df.head(100),

            use_container_width=True,

        )


        # Try to identify simulation paths

        if isinstance(
            simulation_result,
            np.ndarray,
        ):

            simulation_array = (
                simulation_result
            )

        elif isinstance(
            simulation_result,
            pd.DataFrame,
        ):

            simulation_array = (
                simulation_result.values
            )

        else:

            simulation_array = None


        if (
            simulation_array is not None
            and simulation_array.ndim == 2
        ):

            simulation_chart = go.Figure()


            number_of_paths = min(
                100,
                simulation_array.shape[1],
            )


            for i in range(
                number_of_paths
            ):

                simulation_chart.add_trace(

                    go.Scatter(

                        y=simulation_array[:, i],

                        mode="lines",

                        line={
                            "width": 1
                        },

                        showlegend=False,

                    )

                )


            simulation_chart.update_layout(

                title="Simulated Portfolio Paths",

                xaxis_title="Day",

                yaxis_title="Portfolio Value",

                height=600,

            )


            st.plotly_chart(

                simulation_chart,

                use_container_width=True,

            )


# ============================================================
# OPTIMIZATION TAB
# ============================================================

with tab_optimization:

    st.header("Portfolio Optimization")

    optimization_method = st.selectbox(
        "Optimization method",
        [
            "Equal Weight",
            "Inverse Volatility",
            "Minimum Volatility",
            "Maximum Sharpe",
            "Risk Parity",
            "Maximum Diversification",
            "Target Volatility",
        ],
        key="optimization_method_selector",
    )

    target_volatility_value = 0.40

    if optimization_method == "Target Volatility":
        target_volatility_value = st.slider(
            "Target annual volatility",
            min_value=0.05,
            max_value=1.00,
            value=0.40,
            step=0.05,
        )

    st.button(
        "Calculate Optimal Weights",
        use_container_width=True,
        on_click=run_optimization_callback,
        args=(
            returns,
            tuple(available_coin_ids),
            optimization_method,
            target_volatility_value,
        ),
    )

    optimization_error = st.session_state.get("optimization_error")

    if optimization_error:
        st.error(f"Optimization failed: {optimization_error}")

    optimization_result = st.session_state.get("optimization_result")
    stored_signature = st.session_state.get("optimization_signature")
    current_signature = (
        tuple(available_coin_ids),
        str(start_date),
        str(end_date),
    )

    if optimization_result is not None:
        if stored_signature != current_signature:
            st.warning(
                "The stored optimized weights are not valid for the "
                "current portfolio/date range."
            )
        else:
            optimized_array = np.asarray(
                optimization_result,
                dtype=float,
            ).reshape(-1)

            if len(optimized_array) == len(returns.columns):
                optimized_array = optimized_array / optimized_array.sum()

                optimization_df = pd.DataFrame({
                    "Cryptocurrency": [
                        coin_name(coin_id) for coin_id in returns.columns
                    ],
                    "CoinGecko ID": list(returns.columns),
                    "Optimized Weight": optimized_array,
                })

                st.success(
                    "Optimization completed. Risk & Performance is using "
                    "these optimized weights."
                )

                st.subheader("Optimized Portfolio Weights")

                st.dataframe(
                    optimization_df.style.format({
                        "Optimized Weight": "{:.2%}"
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=optimization_df["Cryptocurrency"],
                        y=optimization_df["Optimized Weight"],
                        name="Optimized Weight",
                    )
                )
                fig.update_layout(
                    title=(
                        f"{st.session_state.get('optimization_method', optimization_method)} "
                        "Portfolio Allocation"
                    ),
                    xaxis_title="Cryptocurrency",
                    yaxis_title="Weight",
                    yaxis_tickformat=".0%",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# DATA TAB
# ============================================================

with tab_data:

    st.header(
        "Market Data"
    )


    st.subheader(
        "Top 100 Cryptocurrencies"
    )


    display_columns = [

        column

        for column in [

            "Rank",

            "Name",

            "Symbol",

            "Price",

            "Market Cap",

            "24h Change",

            "7d Change",

            "30d Change",

        ]

        if column in top_coins.columns

    ]


    st.dataframe(

        top_coins[
            display_columns
        ],

        use_container_width=True,

        height=600,

    )


    st.subheader(
        "Historical Price Data"
    )


    display_prices = prices_df.copy()


    display_prices.columns = [

        coin_name(coin_id)

        for coin_id
        in display_prices.columns

    ]


    st.dataframe(

        display_prices.tail(100),

        use_container_width=True,

    )


# ============================================================
# AI ANALYSIS TAB
# ============================================================

with tab_ai:

    st.header(
        "🤖 AI Portfolio Analysis"
    )


    st.write(
        "Generate an AI-based summary of the "
        "current portfolio, risk metrics and "
        "optimization results."
    )


    if st.button(
        "Generate AI Analysis"
    ):

        portfolio_context = {

            "selected_coins": [

                coin_name(coin_id)

                for coin_id
                in selected_coin_ids

            ],

            "weights": {

                coin_name(coin_id):

                float(weight)

                for coin_id, weight
                in zip(
                    selected_coin_ids,
                    weights,
                )

            },

            "start_date":
                str(start_date),

            "end_date":
                str(end_date),

            "portfolio_return":
                float(risk_portfolio_value.iloc[-1] - 1),

            "annual_return":
                float(annual_return),

            "annual_volatility":
                float(annual_volatility),

            "sharpe_ratio":
                float(sharpe)
                if pd.notna(sharpe)
                else None,

            "sortino_ratio":
                float(sortino)
                if pd.notna(sortino)
                else None,

            "var_95":
                float(var_95),

            "cvar_95":
                float(cvar_95),

            "max_drawdown":
                float(max_drawdown),

        }


        # Add Monte Carlo result if available

        if (
            "monte_carlo_result"
            in st.session_state
        ):

            portfolio_context[
                "monte_carlo"
            ] = str(

                st.session_state[
                    "monte_carlo_result"
                ]

            )[:5000]


        # Add optimization result

        if (
            "optimization_result"
            in st.session_state
        ):

            portfolio_context[
                "optimization"
            ] = str(

                st.session_state[
                    "optimization_result"
                ]

            )[:5000]


        try:

            ai_result = generate_ai_analysis(

                portfolio_context

            )


            st.subheader(
                "AI Analysis"
            )


            st.write(
                ai_result
            )


        except Exception as e:

            st.error(
                f"AI analysis failed: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Crypto Portfolio & Risk Analysis"
)

st.sidebar.caption(
    "Historical market data: CoinGecko"
)

# ============================================================
# DOCUMENTATION TAB
# ============================================================

with tab_documentation:

    render_documentation()