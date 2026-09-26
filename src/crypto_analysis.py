import json
import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = os.path.join(
    "data",
    "crypto_research",
)


# ============================================================
# LOCAL CACHE
# ============================================================

def ensure_data_directory():
    os.makedirs(
        DATA_DIR,
        exist_ok=True,
    )


def get_cache_path(coin_id):
    ensure_data_directory()

    return os.path.join(
        DATA_DIR,
        f"{coin_id}.json",
    )


def save_research_data(
    coin_id,
    data,
):
    path = get_cache_path(
        coin_id
    )

    payload = {
        "cached_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "coin_id": coin_id,

        "data": data,
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def load_research_data(
    coin_id,
):
    path = get_cache_path(
        coin_id
    )

    if not os.path.exists(path):
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(
                file
            )

        return payload

    except Exception:
        return None


# ============================================================
# COIN SYMBOL
# ============================================================

COIN_SYMBOLS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "tether": "USDT-USD",
    "binancecoin": "BNB-USD",
    "solana": "SOL-USD",
    "usd-coin": "USDC-USD",
    "xrp": "XRP-USD",
    "dogecoin": "DOGE-USD",
    "cardano": "ADA-USD",
    "avalanche-2": "AVAX-USD",
    "chainlink": "LINK-USD",
    "sui": "SUI-USD",
    "polkadot": "DOT-USD",
    "tron": "TRX-USD",
    "uniswap": "UNI-USD",
    "near": "NEAR-USD",
    "litecoin": "LTC-USD",
    "internet-computer": "ICP-USD",
    "aptos": "APT-USD",
    "cosmos": "ATOM-USD",
    "stellar": "XLM-USD",
    "hedera-hashgraph": "HBAR-USD",
    "filecoin": "FIL-USD",
    "arbitrum": "ARB-USD",
    "optimism": "OP-USD",
    "aave": "AAVE-USD",
    "maker": "MKR-USD",
    "the-graph": "GRT-USD",
    "algorand": "ALGO-USD",
    "vechain": "VET-USD",
}


def get_yahoo_symbol(
    coin_id,
):
    return COIN_SYMBOLS.get(
        coin_id
    )


# ============================================================
# YAHOO FINANCE DATA
# ============================================================

def get_yahoo_market_data(
    coin_id,
):
    """
    Download market and historical price data
    from Yahoo Finance.

    This does not use CoinGecko.
    """

    ticker_symbol = get_yahoo_symbol(
        coin_id
    )

    if not ticker_symbol:

        raise ValueError(
            f"No Yahoo Finance symbol configured "
            f"for {coin_id}."
        )

    ticker = yf.Ticker(
        ticker_symbol
    )

    history = ticker.history(
        period="1y",
        interval="1d",
        auto_adjust=False,
    )

    if history.empty:

        raise ValueError(
            f"No Yahoo Finance price data "
            f"available for {ticker_symbol}."
        )

    prices = (
        history["Close"]
        .dropna()
    )

    current_price = float(
        prices.iloc[-1]
    )

    returns = (
        prices
        .pct_change()
        .dropna()
    )

    market_data = {
        "ticker": ticker_symbol,

        "current_price": current_price,

        "price_7d": (
            float(prices.iloc[-8])
            if len(prices) > 7
            else None
        ),

        "price_30d": (
            float(prices.iloc[-31])
            if len(prices) > 30
            else None
        ),

        "price_90d": (
            float(prices.iloc[-91])
            if len(prices) > 90
            else None
        ),

        "price_1y": (
            float(prices.iloc[0])
            if len(prices) > 1
            else None
        ),

        "historical_high_1y": float(
            prices.max()
        ),

        "historical_low_1y": float(
            prices.min()
        ),

        "annualized_volatility": (
            float(
                returns.std()
                * (365 ** 0.5)
            )
            if len(returns) > 1
            else None
        ),

        "maximum_drawdown": calculate_max_drawdown(
            prices
        ),

        "volume_latest": (
            float(
                history["Volume"]
                .iloc[-1]
            )
            if "Volume" in history.columns
            else None
        ),
    }

    return {
        "market": market_data,
        "prices": prices,
    }


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    prices,
):
    prices = pd.Series(
        prices
    ).dropna()

    if prices.empty:
        return None

    rolling_max = prices.cummax()

    drawdown = (
        prices / rolling_max
    ) - 1

    return float(
        drawdown.min()
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

PROJECT_DATA = {

    "bitcoin": {
        "name": "Bitcoin",
        "symbol": "BTC",
        "type": "Layer 1 / Cryptocurrency",
        "consensus": "Proof of Work",
        "description": (
            "A decentralized digital currency "
            "secured through a proof-of-work network."
        ),
    },

    "ethereum": {
        "name": "Ethereum",
        "symbol": "ETH",
        "type": "Layer 1 / Smart Contract Platform",
        "consensus": "Proof of Stake",
        "description": (
            "A decentralized blockchain platform "
            "for smart contracts and decentralized applications."
        ),
    },

    "solana": {
        "name": "Solana",
        "symbol": "SOL",
        "type": "Layer 1",
        "consensus": "Proof of Stake / Proof of History",
        "description": (
            "A high-throughput blockchain designed "
            "for decentralized applications and digital assets."
        ),
    },

    "cardano": {
        "name": "Cardano",
        "symbol": "ADA",
        "type": "Layer 1",
        "consensus": "Proof of Stake",
        "description": (
            "A proof-of-stake blockchain focused "
            "on research-driven development."
        ),
    },

    "sui": {
        "name": "Sui",
        "symbol": "SUI",
        "type": "Layer 1",
        "consensus": "Delegated Proof of Stake",
        "description": (
            "A Layer 1 blockchain designed for "
            "high-throughput decentralized applications."
        ),
    },

    "avalanche-2": {
        "name": "Avalanche",
        "symbol": "AVAX",
        "type": "Layer 1",
        "consensus": "Proof of Stake",
        "description": (
            "A blockchain platform supporting "
            "custom networks and decentralized applications."
        ),
    },

    "chainlink": {
        "name": "Chainlink",
        "symbol": "LINK",
        "type": "Oracle Network",
        "consensus": "Decentralized Oracle Network",
        "description": (
            "A decentralized oracle infrastructure "
            "connecting blockchains with external data."
        ),
    },
}


def get_project_data(
    coin_id,
):
    return PROJECT_DATA.get(
        coin_id,
        {
            "name": coin_id,
            "symbol": "",
            "type": "Unknown",
            "consensus": "Unknown",
            "description": (
                "Project information "
                "has not yet been collected."
            ),
        },
    )


# ============================================================
# FULL RESEARCH
# ============================================================

def refresh_coin_research(
    coin_id,
):
    """
    Refresh research data.

    External market data comes from Yahoo Finance.
    Project information comes from our local
    research configuration.

    Additional research sources will be added
    in the next stages.
    """

    yahoo_data = get_yahoo_market_data(
        coin_id
    )

    project_data = get_project_data(
        coin_id
    )

    prices = yahoo_data[
        "prices"
    ]

    research = {
        "coin": {
            "id": coin_id,

            "name": project_data.get(
                "name"
            ),

            "symbol": project_data.get(
                "symbol"
            ),
        },

        "market": yahoo_data[
            "market"
        ],

        "project": project_data,

        "tokenomics": {},

        "investors": {},

        "funding": {},

        "unlocks": {},

        "history": {},

        "ecosystem": {},

        "github": {},

        "events": {},

        "risks": {},

        "price_history": {
            str(index.date()): float(value)
            for index, value
            in prices.items()
        },
    }

    save_research_data(
        coin_id,
        research,
    )

    return load_research_data(
        coin_id
    )


# ============================================================
# LOAD OR REFRESH
# ============================================================

def get_coin_research(
    coin_id,
    refresh=False,
):
    """
    Use cached research unless refresh=True.
    """

    if not refresh:

        cached = load_research_data(
            coin_id
        )

        if cached is not None:

            return cached

    return refresh_coin_research(
        coin_id
    )


# ============================================================
# PRICE TREND ANALYSIS
# ============================================================

def calculate_price_trends(
    prices,
):
    prices = pd.Series(
        prices
    ).dropna()

    if len(prices) < 2:
        return {}

    current_price = float(
        prices.iloc[-1]
    )

    result = {
        "current_price": current_price,
    }

    periods = {
        "7D": 7,
        "30D": 30,
        "90D": 90,
        "180D": 180,
        "1Y": 365,
    }

    for name, days in periods.items():

        if len(prices) > days:

            old_price = float(
                prices.iloc[-days - 1]
            )

            if old_price != 0:

                result[
                    f"{name}_return"
                ] = (
                    current_price /
                    old_price
                ) - 1

            else:

                result[
                    f"{name}_return"
                ] = None

        else:

            result[
                f"{name}_return"
            ] = None

    historical_ath = float(
        prices.max()
    )

    historical_atl = float(
        prices.min()
    )

    result[
        "historical_ath"
    ] = historical_ath

    result[
        "historical_atl"
    ] = historical_atl

    if historical_ath != 0:

        result[
            "distance_from_ath"
        ] = (
            current_price /
            historical_ath
        ) - 1

    if historical_atl != 0:

        result[
            "distance_from_atl"
        ] = (
            current_price /
            historical_atl
        ) - 1

    result[
        "maximum_drawdown"
    ] = calculate_max_drawdown(
        prices
    )

    returns = (
        prices
        .pct_change()
        .dropna()
    )

    if len(returns) > 1:

        result[
            "annualized_volatility"
        ] = float(
            returns.std()
            * (365 ** 0.5)
        )

    else:

        result[
            "annualized_volatility"
        ] = None

    return result