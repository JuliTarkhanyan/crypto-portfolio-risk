# ============================================================
# SRC/SINGLE_CRYPTO_ANALYSIS.PY
# ============================================================

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import (
    download_coin_details_by_id,
    download_market_chart_by_id,
    download_top_100_historical_prices,
    download_defillama_protocol,
    download_defillama_protocols,
)


# ============================================================
# SINGLE CRYPTO RESEARCH HELPERS
# ============================================================

def _safe_float(value):
    try:
        if value is None:
            return np.nan

        return float(value)

    except (
        ValueError,
        TypeError,
    ):
        return np.nan


def _format_money(value):
    if pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"

    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"${value / 1_000:.2f}K"

    return f"${value:,.4f}"


def _format_supply(value):
    if pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:,.2f}"


def _format_pct(value):
    if pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.2f}%"


def _clean_html(text):
    if not text:
        return ""

    return (
        re.sub(
            r"<[^>]+>",
            " ",
            str(text),
        )
        .replace(
            "&amp;",
            "&",
        )
        .replace(
            "&nbsp;",
            " ",
        )
        .strip()
    )


def _calculate_rsi(
    prices,
    period=14,
):
    returns = prices.pct_change().dropna()

    if len(returns) < period:
        return np.nan

    gains = returns.clip(
        lower=0
    )

    losses = -returns.clip(
        upper=0
    )

    avg_gain = gains.rolling(
        period
    ).mean()

    avg_loss = losses.rolling(
        period
    ).mean()

    if pd.isna(
        avg_loss.iloc[-1]
    ):
        return np.nan

    if avg_loss.iloc[-1] == 0:
        return 100.0

    rs = (
        avg_gain.iloc[-1]
        / avg_loss.iloc[-1]
    )

    return 100 - (
        100 / (1 + rs)
    )


def _period_return(
    prices,
    days,
):
    prices = prices.dropna()

    if len(prices) < 2:
        return np.nan

    start_index = max(
        0,
        len(prices) - days - 1,
    )

    start_price = prices.iloc[
        start_index
    ]

    end_price = prices.iloc[-1]

    if start_price == 0:
        return np.nan

    return (
        end_price / start_price
        - 1
    )


def _trend(prices):
    prices = prices.dropna()

    if len(prices) < 200:

        if len(prices) >= 50:

            sma_20 = (
                prices
                .rolling(20)
                .mean()
                .iloc[-1]
            )

            sma_50 = (
                prices
                .rolling(50)
                .mean()
                .iloc[-1]
            )

            current = prices.iloc[-1]

            if current > sma_20 > sma_50:
                return "Upward trend"

            if current < sma_20 < sma_50:
                return "Downward trend"

        return "Insufficient / mixed data"

    current = prices.iloc[-1]

    sma_50 = (
        prices
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    sma_200 = (
        prices
        .rolling(200)
        .mean()
        .iloc[-1]
    )

    if current > sma_50 > sma_200:
        return "Strong upward trend"

    if current < sma_50 < sma_200:
        return "Downward trend"

    return "Mixed / sideways"


def _drawdown(prices):
    running_max = prices.cummax()

    return (
        prices / running_max
        - 1
    )


# ============================================================
# DATA LOADERS
# ============================================================

def load_single_coin_details(
    coin_id,
    force_refresh=False,
):
    """
    Load cryptocurrency details.

    Persistent caching is handled by src.data.
    """

    try:

        return download_coin_details_by_id(
            coin_id,
            force_refresh=force_refresh,
        )

    except Exception as e:

        return {
            "error": str(e)
        }


def load_single_coin_history(
    coin_id,
    days=365,
    force_refresh=False,
):
    """
    Load historical prices for one cryptocurrency.

    PRIMARY SOURCE:
        Persistent Top 100 historical-price cache.

    FALLBACK:
        Individual CoinGecko market-chart endpoint.

    This prevents Single Crypto Research from
    unnecessarily requesting separate historical
    data when the cryptocurrency already exists
    in the application's Top 100 historical cache.
    """

    # ========================================================
    # PRIMARY SOURCE — TOP 100 CACHE
    # ========================================================

    try:

        all_prices = (
            download_top_100_historical_prices(
                days=days,
                force_refresh=force_refresh,
            )
        )

        if (
            all_prices is not None
            and not all_prices.empty
        ):

            # ------------------------------------------------
            # Normalize possible column naming
            # ------------------------------------------------

            if coin_id in all_prices.columns:

                prices = (
                    all_prices[coin_id]
                    .dropna()
                    .copy()
                )

                if not prices.empty:

                    prices.index = pd.to_datetime(
                        prices.index,
                        errors="coerce",
                    )

                    prices = prices[
                        ~prices.index.isna()
                    ]

                    if (
                        getattr(
                            prices.index,
                            "tz",
                            None,
                        )
                        is not None
                    ):

                        prices.index = (
                            prices.index.tz_localize(
                                None
                            )
                        )

                    prices = (
                        pd.to_numeric(
                            prices,
                            errors="coerce",
                        )
                        .dropna()
                        .sort_index()
                    )

                    if not prices.empty:

                        return prices.to_frame(
                            name=coin_id
                        )

    except Exception:
        pass

    # ========================================================
    # FALLBACK — INDIVIDUAL MARKET CHART
    # ========================================================

    try:

        history = (
            download_market_chart_by_id(
                coin_id,
                days=days,
                force_refresh=force_refresh,
            )
        )

        return history

    except Exception:

        return pd.DataFrame()


def load_single_coin_btc_history(
    force_refresh=False,
):
    """
    Load Bitcoin history using the same
    persistent Top 100 historical cache.
    """

    return load_single_coin_history(
        coin_id="bitcoin",
        days=365,
        force_refresh=force_refresh,
    )


def load_defillama_protocols(
    force_refresh=False,
):
    """
    Load DeFiLlama protocol list.

    Persistent caching is handled by src.data.
    """

    try:

        return download_defillama_protocols(
            force_refresh=force_refresh,
        )

    except Exception:

        return []


def load_defillama_protocol(
    slug,
    force_refresh=False,
):
    """
    Load DeFiLlama protocol details.

    Persistent caching is handled by src.data.
    """

    try:

        return download_defillama_protocol(
            slug,
            force_refresh=force_refresh,
        )

    except Exception:

        return None


# ============================================================
# HISTORY EXTRACTION
# ============================================================

def _extract_coin_prices_from_history(
    history,
):
    """
    Accept:

        pandas Series
        pandas DataFrame
        old CoinGecko dictionary format

    Returns:

        pandas Series
    """

    # ========================================================
    # PANDAS SERIES
    # ========================================================

    if isinstance(
        history,
        pd.Series,
    ):

        if history.empty:

            return pd.Series(
                dtype=float
            )

        series = history.copy()

        series.index = pd.to_datetime(
            series.index,
            errors="coerce",
        )

        series = series[
            ~series.index.isna()
        ]

        if (
            getattr(
                series.index,
                "tz",
                None,
            )
            is not None
        ):

            series.index = (
                series.index.tz_localize(
                    None
                )
            )

        return (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .dropna()
            .sort_index()
        )

    # ========================================================
    # PANDAS DATAFRAME
    # ========================================================

    if isinstance(
        history,
        pd.DataFrame,
    ):

        if history.empty:

            return pd.Series(
                dtype=float
            )

        if history.shape[1] == 0:

            return pd.Series(
                dtype=float
            )

        series = (
            history.iloc[:, 0]
            .copy()
        )

        series.index = pd.to_datetime(
            series.index,
            errors="coerce",
        )

        series = series[
            ~series.index.isna()
        ]

        if (
            getattr(
                series.index,
                "tz",
                None,
            )
            is not None
        ):

            series.index = (
                series.index.tz_localize(
                    None
                )
            )

        return (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .dropna()
            .sort_index()
        )

    # ========================================================
    # OLD COINGECKO DICTIONARY FORMAT
    # ========================================================

    if not isinstance(
        history,
        dict,
    ):

        return pd.Series(
            dtype=float
        )

    raw_prices = history.get(
        "prices",
        [],
    )

    if not raw_prices:

        return pd.Series(
            dtype=float
        )

    data = pd.DataFrame(
        raw_prices,
        columns=[
            "timestamp",
            "price",
        ],
    )

    data["date"] = pd.to_datetime(
        data["timestamp"],
        unit="ms",
        errors="coerce",
    )

    data = data[
        ~data["date"].isna()
    ]

    data = (
        data
        .drop_duplicates(
            "date"
        )
        .set_index(
            "date"
        )
    )

    return (
        pd.to_numeric(
            data["price"],
            errors="coerce",
        )
        .dropna()
        .sort_index()
    )


# ============================================================
# BTC CORRELATION
# ============================================================

def _calculate_btc_correlation(
    prices,
    force_refresh=False,
):
    btc_history = (
        load_single_coin_btc_history(
            force_refresh=force_refresh
        )
    )

    btc_prices = (
        _extract_coin_prices_from_history(
            btc_history
        )
    )

    if (
        prices.empty
        or btc_prices.empty
    ):

        return np.nan

    coin_returns = (
        prices.pct_change()
    )

    btc_returns = (
        btc_prices.pct_change()
    )

    combined = pd.concat(
        [
            coin_returns.rename(
                "coin"
            ),
            btc_returns.rename(
                "btc"
            ),
        ],
        axis=1,
    ).dropna()

    if len(combined) < 30:

        return np.nan

    return combined[
        "coin"
    ].corr(
        combined["btc"]
    )


# ============================================================
# DEFILLAMA MATCH
# ============================================================

def _find_defillama_protocol(
    name,
    symbol,
    force_refresh=False,
):
    protocols = (
        load_defillama_protocols(
            force_refresh=force_refresh
        )
    )

    if not protocols:

        return None

    name = str(
        name
    ).lower()

    symbol = str(
        symbol
    ).lower()

    for protocol in protocols:

        protocol_name = str(
            protocol.get(
                "name",
                "",
            )
        ).lower()

        protocol_symbol = str(
            protocol.get(
                "symbol",
                "",
            )
        ).lower()

        if protocol_name == name:

            return protocol

        if (
            protocol_symbol
            and protocol_symbol == symbol
        ):

            return protocol

    return None


# ============================================================
# MAIN RENDER FUNCTION
# ============================================================

def render_single_crypto_research(
    coin_id,
    coin_name,
    coin_symbol,
    top_coins,
    portfolio_prices_df=None,
    refresh=False,
):
    """
    Render the complete Single Crypto Research tab.

    Cache behavior:

    refresh=False:
        Use persistent cache whenever available.

    refresh=True:
        Request fresh data through src.data and update
        the persistent cache.

    The refresh is limited to this research workflow.
    """

    # ========================================================
    # LOAD COIN DETAILS
    # ========================================================

    with st.spinner(
        f"Loading research for {coin_name}..."
    ):

        details = (
            load_single_coin_details(
                coin_id,
                force_refresh=refresh,
            )
        )

    # ========================================================
    # FALLBACK TO TOP 100 MARKET DATA
    # ========================================================

    if (
        not details
        or "error" in details
    ):

        detail_error = (
            details.get(
                "error",
                "Unknown error",
            )
            if isinstance(
                details,
                dict,
            )
            else "Unknown error"
        )

        fallback_row = None

        if isinstance(
            top_coins,
            pd.DataFrame,
        ):

            if (
                "CoinGecko ID"
                in top_coins.columns
            ):

                matches = top_coins[
                    top_coins[
                        "CoinGecko ID"
                    ]
                    .astype(str)
                    == str(coin_id)
                ]

            elif "id" in top_coins.columns:

                matches = top_coins[
                    top_coins["id"]
                    .astype(str)
                    == str(coin_id)
                ]

            else:

                matches = pd.DataFrame()

            if not matches.empty:

                fallback_row = (
                    matches.iloc[0]
                )

        if fallback_row is None:

            st.error(
                "Could not load detailed "
                "cryptocurrency information "
                "and no cached market data is "
                "available for this coin."
            )

            st.caption(
                detail_error
            )

            return

        details = {
            "market_cap_rank": (
                fallback_row.get(
                    "Rank"
                )
            ),
            "description": {
                "en": ""
            },
            "categories": [],
            "links": {},
            "market_data": {
                "market_cap": {
                    "usd": fallback_row.get(
                        "Market Cap"
                    )
                },
                "fully_diluted_valuation": {
                    "usd": np.nan
                },
                "total_volume": {
                    "usd": fallback_row.get(
                        "24h Volume"
                    )
                },
                "circulating_supply": np.nan,
                "total_supply": np.nan,
                "max_supply": np.nan,
            },
        }

        st.warning(
            "Detailed CoinGecko information "
            "is unavailable. The research is "
            "continuing with cached market data."
        )

    # ========================================================
    # LOAD HISTORICAL DATA
    # ========================================================

    with st.spinner(
        f"Loading historical prices for {coin_name}..."
    ):

        history = (
            load_single_coin_history(
                coin_id,
                days=365,
                force_refresh=refresh,
            )
        )

    prices = (
        _extract_coin_prices_from_history(
            history
        )
        .dropna()
        .sort_index()
    )

    # ========================================================
    # PORTFOLIO DATA FALLBACK
    # ========================================================

    if (
        prices.empty
        and portfolio_prices_df is not None
    ):

        try:

            if (
                coin_id
                in portfolio_prices_df.columns
            ):

                prices = (
                    portfolio_prices_df[
                        coin_id
                    ]
                    .dropna()
                    .sort_index()
                )

        except Exception:

            pass

    # ========================================================
    # FINAL HISTORY CHECK
    # ========================================================

    if prices.empty:

        st.error(
            f"Historical data for {coin_name} "
            f"({coin_symbol}) could not be loaded."
        )

        st.info(
            "The application checked the persistent "
            "Top 100 historical cache and the individual "
            "market-chart source, but neither returned "
            "historical prices."
        )

        st.caption(
            "If the Portfolio tab already shows historical "
            "prices for this cryptocurrency, the issue is "
            "most likely the CoinGecko ID/column mapping "
            "inside src/data.py."
        )

        return

    # ========================================================
    # NORMALIZE TIMEZONE
    # ========================================================

    prices.index = pd.to_datetime(
        prices.index,
        errors="coerce",
    )

    prices = prices[
        ~prices.index.isna()
    ]

    if (
        getattr(
            prices.index,
            "tz",
            None,
        )
        is not None
    ):

        prices.index = (
            prices.index.tz_localize(
                None
            )
        )

    prices = (
        pd.to_numeric(
            prices,
            errors="coerce",
        )
        .dropna()
        .sort_index()
    )

    if prices.empty:

        st.error(
            "Historical data was returned, "
            "but it contained no valid numeric "
            "price observations."
        )

        return

    # ========================================================
    # AVAILABLE RANGE
    # ========================================================

    available_start = (
        prices.index.min()
    )

    available_end = (
        prices.index.max()
    )

    st.info(
        f"📅 **Available historical range:** "
        f"{available_start:%Y-%m-%d} → "
        f"{available_end:%Y-%m-%d} "
        f"| **Observations:** "
        f"{len(prices):,}"
    )

    # ========================================================
    # ANALYSIS PERIOD
    # ========================================================

    period_options = {
        "7D": 7,
        "30D": 30,
        "90D": 90,
        "1Y": 365,
        "Max available": None,
    }

    selected_period = st.selectbox(
        "Analysis period",
        options=list(
            period_options.keys()
        ),
        index=(
            3
            if len(prices) > 365
            else len(period_options) - 1
        ),
        key=(
            f"single_crypto_period_"
            f"{coin_id}"
        ),
    )

    period_days = (
        period_options[
            selected_period
        ]
    )

    if period_days is None:

        analysis_prices = (
            prices.copy()
        )

    else:

        requested_start = (
            available_end
            - pd.Timedelta(
                days=period_days
            )
        )

        analysis_prices = prices.loc[
            prices.index
            >= requested_start
        ].copy()

    if len(analysis_prices) < 2:

        st.warning(
            "Not enough historical observations "
            f"are available for {selected_period}. "
            "Choose a longer period."
        )

        return

    prices = analysis_prices

    # ========================================================
    # CALCULATIONS
    # ========================================================

    returns_single = (
        prices
        .pct_change()
        .dropna()
    )

    current_price = (
        prices.iloc[-1]
    )

    return_7d = _period_return(
        prices,
        7,
    )

    return_30d = _period_return(
        prices,
        30,
    )

    return_90d = _period_return(
        prices,
        90,
    )

    return_1y = _period_return(
        prices,
        365,
    )

    volatility = (
        returns_single.std()
        * np.sqrt(365)
    )

    maximum_dd = (
        _drawdown(
            prices
        ).min()
    )

    rsi = _calculate_rsi(
        prices
    )

    trend = _trend(
        prices
    )

    btc_correlation = (
        _calculate_btc_correlation(
            prices,
            force_refresh=refresh,
        )
    )

    # ========================================================
    # MARKET DATA
    # ========================================================

    market_data = details.get(
        "market_data",
        {},
    )

    market_cap = _safe_float(
        market_data.get(
            "market_cap",
            {},
        ).get(
            "usd"
        )
    )

    fdv = _safe_float(
        market_data.get(
            "fully_diluted_valuation",
            {},
        ).get(
            "usd"
        )
    )

    volume = _safe_float(
        market_data.get(
            "total_volume",
            {},
        ).get(
            "usd"
        )
    )

    circulating_supply = _safe_float(
        market_data.get(
            "circulating_supply"
        )
    )

    total_supply = _safe_float(
        market_data.get(
            "total_supply"
        )
    )

    max_supply = _safe_float(
        market_data.get(
            "max_supply"
        )
    )

    market_rank = details.get(
        "market_cap_rank"
    )

    mc_fdv = (
        market_cap / fdv
        if (
            not pd.isna(fdv)
            and fdv > 0
        )
        else np.nan
    )

    circulating_ratio = (
        circulating_supply
        / max_supply
        if (
            not pd.isna(
                circulating_supply
            )
            and not pd.isna(
                max_supply
            )
            and max_supply > 0
        )
        else np.nan
    )

    # ========================================================
    # HEADER
    # ========================================================

    st.header(
        f"🔬 {coin_name} "
        f"({coin_symbol})"
    )

    st.caption(
        "Single-cryptocurrency research based on "
        "market data, price tendencies, technical "
        "indicators, tokenomics, project information "
        "and DeFi data."
    )

    # ========================================================
    # CACHE STATUS
    # ========================================================

    if refresh:

        st.success(
            "🔄 Selected cryptocurrency data was "
            "refreshed through the data layer."
        )

    else:

        st.caption(
            "💾 Historical data is loaded from the "
            "persistent local cache whenever available."
        )

    # ========================================================
    # TOP METRICS
    # ========================================================

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:

        st.metric(
            "Price",
            _format_money(
                current_price
            ),
        )

    with col2:

        st.metric(
            "Market Cap",
            _format_money(
                market_cap
            ),
        )

    with col3:

        st.metric(
            "FDV",
            _format_money(
                fdv
            ),
        )

    with col4:

        st.metric(
            "24h Volume",
            _format_money(
                volume
            ),
        )

    with col5:

        st.metric(
            "Market Rank",
            (
                f"#{int(market_rank)}"
                if pd.notna(
                    market_rank
                )
                else "N/A"
            ),
        )

    # ========================================================
    # RESEARCH SUB-TABS
    # ========================================================

    (
        research_overview,
        research_technical,
        research_tokenomics,
        research_project,
        research_defi,
        research_summary,
    ) = st.tabs(
        [
            "📊 Overview",
            "📈 Technical",
            "🪙 Tokenomics",
            "🏗️ Project",
            "🌐 DeFi",
            "📝 Research Summary",
        ]
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    with research_overview:

        st.subheader(
            "Market Overview"
        )

        overview_df = pd.DataFrame(
            {
                "Metric": [
                    "7D Return",
                    "30D Return",
                    "90D Return",
                    "1Y Return",
                    "Market Cap / FDV",
                    "BTC Correlation",
                    "Annualized Volatility",
                    "Maximum Drawdown",
                ],
                "Value": [
                    _format_pct(
                        return_7d
                    ),
                    _format_pct(
                        return_30d
                    ),
                    _format_pct(
                        return_90d
                    ),
                    _format_pct(
                        return_1y
                    ),
                    _format_pct(
                        mc_fdv
                    ),
                    (
                        f"{btc_correlation:.3f}"
                        if pd.notna(
                            btc_correlation
                        )
                        else "N/A"
                    ),
                    _format_pct(
                        volatility
                    ),
                    _format_pct(
                        maximum_dd
                    ),
                ],
            }
        )

        st.dataframe(
            overview_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Price Trend"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices.values,
                mode="lines",
                name=coin_symbol,
            )
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # TECHNICAL
    # ========================================================

    with research_technical:

        st.subheader(
            "Price Tendencies & "
            "Technical Indicators"
        )

        technical_cols = st.columns(4)

        with technical_cols[0]:

            st.metric(
                "Trend",
                trend,
            )

        with technical_cols[1]:

            st.metric(
                "RSI (14)",
                (
                    f"{rsi:.2f}"
                    if pd.notna(rsi)
                    else "N/A"
                ),
            )

        with technical_cols[2]:

            st.metric(
                "30D Return",
                _format_pct(
                    return_30d
                ),
            )

        with technical_cols[3]:

            st.metric(
                "Max Drawdown",
                _format_pct(
                    maximum_dd
                ),
            )

        technical_chart = go.Figure()

        technical_chart.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices.values,
                mode="lines",
                name="Price",
            )
        )

        sma_20 = (
            prices
            .rolling(20)
            .mean()
        )

        sma_50 = (
            prices
            .rolling(50)
            .mean()
        )

        technical_chart.add_trace(
            go.Scatter(
                x=prices.index,
                y=sma_20,
                mode="lines",
                name="20D SMA",
            )
        )

        technical_chart.add_trace(
            go.Scatter(
                x=prices.index,
                y=sma_50,
                mode="lines",
                name="50D SMA",
            )
        )

        if len(prices) >= 200:

            sma_200 = (
                prices
                .rolling(200)
                .mean()
            )

            technical_chart.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=sma_200,
                    mode="lines",
                    name="200D SMA",
                )
            )

        technical_chart.update_layout(
            title="Price and Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=550,
            hovermode="x unified",
        )

        st.plotly_chart(
            technical_chart,
            use_container_width=True,
        )

        st.subheader(
            "Drawdown"
        )

        drawdown = _drawdown(
            prices
        )

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
            height=350,
            hovermode="x unified",
        )

        st.plotly_chart(
            drawdown_chart,
            use_container_width=True,
        )

        st.info(
            "Technical indicators describe "
            "historical price behavior. They do "
            "not guarantee future price movements."
        )

    # ========================================================
    # TOKENOMICS
    # ========================================================

    with research_tokenomics:

        st.subheader(
            "Tokenomics"
        )

        tokenomics_df = pd.DataFrame(
            {
                "Metric": [
                    "Circulating Supply",
                    "Total Supply",
                    "Maximum Supply",
                    "Circulating / Max Supply",
                    "Market Cap / FDV",
                ],
                "Value": [
                    _format_supply(
                        circulating_supply
                    ),
                    _format_supply(
                        total_supply
                    ),
                    _format_supply(
                        max_supply
                    ),
                    _format_pct(
                        circulating_ratio
                    ),
                    _format_pct(
                        mc_fdv
                    ),
                ],
            }
        )

        st.dataframe(
            tokenomics_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Token Unlocks"
        )

        st.info(
            "Detailed vesting schedules, future "
            "unlock dates, allocation percentages "
            "and unlock pressure are not inferred "
            "from supply data. Use a dedicated "
            "tokenomics source such as CryptoRank "
            "for those metrics when API access is "
            "available."
        )

    # ========================================================
    # PROJECT
    # ========================================================

    with research_project:

        st.subheader(
            "Project Fundamentals"
        )

        description = _clean_html(
            details.get(
                "description",
                {},
            ).get(
                "en",
                "",
            )
        )

        categories = details.get(
            "categories",
            [],
        )

        links = details.get(
            "links",
            {},
        )

        homepage = [
            x
            for x in links.get(
                "homepage",
                [],
            )
            if x
        ]

        explorers = [
            x
            for x in links.get(
                "blockchain_site",
                [],
            )
            if x
        ]

        github_repos = [
            x
            for x in links.get(
                "repos_url",
                {},
            ).get(
                "github",
                [],
            )
            if x
        ]

        project_cols = st.columns(2)

        with project_cols[0]:

            st.markdown(
                "**Categories**"
            )

            st.write(
                (
                    ", ".join(
                        categories
                    )
                    if categories
                    else "N/A"
                )
            )

            st.markdown(
                "**Website**"
            )

            if homepage:

                st.write(
                    homepage[0]
                )

            else:

                st.write(
                    "N/A"
                )

        with project_cols[1]:

            st.markdown(
                "**Blockchain Explorer**"
            )

            if explorers:

                st.write(
                    explorers[0]
                )

            else:

                st.write(
                    "N/A"
                )

            st.markdown(
                "**GitHub**"
            )

            if github_repos:

                st.write(
                    github_repos[0]
                )

            else:

                st.write(
                    "N/A"
                )

        st.markdown(
            "### Description"
        )

        st.write(
            description
            if description
            else "No project description available."
        )

    # ========================================================
    # DEFI
    # ========================================================

    with research_defi:

        st.subheader(
            "DeFi / On-chain Context"
        )

        protocol = (
            _find_defillama_protocol(
                coin_name,
                coin_symbol,
                force_refresh=refresh,
            )
        )

        if protocol is None:

            st.info(
                "No matching DeFiLlama protocol "
                "was found for this cryptocurrency."
            )

        else:

            protocol_name = protocol.get(
                "name",
                coin_name,
            )

            protocol_slug = protocol.get(
                "slug"
            )

            st.write(
                f"**DeFiLlama match:** "
                f"{protocol_name}"
            )

            protocol_details = None

            if protocol_slug:

                protocol_details = (
                    load_defillama_protocol(
                        protocol_slug,
                        force_refresh=refresh,
                    )
                )

            if protocol_details:

                tvl = _safe_float(
                    protocol_details.get(
                        "tvl"
                    )
                )

                defi_cols = st.columns(4)

                with defi_cols[0]:

                    st.metric(
                        "Current TVL",
                        _format_money(
                            tvl
                        ),
                    )

                with defi_cols[1]:

                    st.metric(
                        "Category",
                        protocol_details.get(
                            "category",
                            "N/A",
                        ),
                    )

                with defi_cols[2]:

                    chains = (
                        protocol_details.get(
                            "chains",
                            [],
                        )
                    )

                    st.metric(
                        "Chains",
                        (
                            str(
                                len(chains)
                            )
                            if chains
                            else "N/A"
                        ),
                    )

                with defi_cols[3]:

                    st.metric(
                        "Protocol",
                        protocol_name,
                    )

                if chains:

                    st.write(
                        "**Chains:** "
                        + ", ".join(
                            chains
                        )
                    )

                protocol_description = (
                    protocol_details.get(
                        "description"
                    )
                )

                if protocol_description:

                    st.write(
                        protocol_description
                    )

                st.caption(
                    "TVL is a protocol-usage metric "
                    "and should be interpreted together "
                    "with price, liquidity, fees, revenue "
                    "and tokenomics."
                )

    # ========================================================
    # RESEARCH SUMMARY
    # ========================================================

    with research_summary:

        st.subheader(
            f"Research Summary — "
            f"{coin_name} ({coin_symbol})"
        )

        st.markdown(
            "### Current Situation"
        )

        st.write(
            f"Based on the selected historical "
            f"period, the current quantitative "
            f"trend is **{trend}**. The 30-day "
            f"return is {_format_pct(return_30d)}, "
            f"while the 1-year return is "
            f"{_format_pct(return_1y)}."
        )

        st.markdown(
            "### Positive Quantitative Factors"
        )

        positive_factors = []

        if (
            pd.notna(return_30d)
            and return_30d > 0
        ):

            positive_factors.append(
                "Positive 30-day price "
                f"performance "
                f"({_format_pct(return_30d)})."
            )

        if (
            pd.notna(return_1y)
            and return_1y > 0
        ):

            positive_factors.append(
                "Positive 1-year price "
                f"performance "
                f"({_format_pct(return_1y)})."
            )

        if (
            pd.notna(rsi)
            and 50 <= rsi < 70
        ):

            positive_factors.append(
                f"RSI is in a moderate range "
                f"({rsi:.1f})."
            )

        if positive_factors:

            for factor in positive_factors:

                st.write(
                    f"• {factor}"
                )

        else:

            st.write(
                "No clear positive quantitative "
                "factor was identified from the "
                "current indicators."
            )

        st.markdown(
            "### Risk / Negative Factors"
        )

        risk_factors = []

        if pd.notna(volatility):

            risk_factors.append(
                "Annualized historical volatility "
                f"is {_format_pct(volatility)}."
            )

        if pd.notna(maximum_dd):

            risk_factors.append(
                "Maximum drawdown over the selected "
                f"period is "
                f"{_format_pct(maximum_dd)}."
            )

        if (
            pd.notna(rsi)
            and rsi >= 70
        ):

            risk_factors.append(
                f"RSI is elevated ({rsi:.1f}), "
                "which can indicate strong recent "
                "momentum and potentially overextended "
                "price conditions."
            )

        if (
            pd.notna(mc_fdv)
            and mc_fdv < 0.50
        ):

            risk_factors.append(
                "Market Cap / FDV is "
                f"{mc_fdv:.2%}, indicating that "
                "a significant portion of fully "
                "diluted value is not represented "
                "by current market cap."
            )

        if risk_factors:

            for factor in risk_factors:

                st.write(
                    f"• {factor}"
                )

        else:

            st.write(
                "No additional quantitative risk "
                "factor was identified from the "
                "available data."
            )

        st.markdown(
            "### What to Monitor"
        )

        monitor = [
            f"Price trend: {trend}.",
            (
                f"RSI: {rsi:.1f}."
                if pd.notna(rsi)
                else "RSI: unavailable."
            ),
            (
                f"BTC correlation: "
                f"{btc_correlation:.3f}."
                if pd.notna(
                    btc_correlation
                )
                else (
                    "BTC correlation: "
                    "unavailable."
                )
            ),
            "Future token unlocks and vesting events.",
            "Changes in trading volume and liquidity.",
            "Project development, ecosystem activity and adoption.",
            "Relevant partnerships, upgrades, listings and regulatory events.",
        ]

        for item in monitor:

            st.write(
                f"• {item}"
            )

        st.info(
            "This is a research dashboard based "
            "on historical and currently available "
            "market/project data. It does not attempt "
            "to predict future prices or provide a "
            "Buy/Hold/Sell recommendation."
        )