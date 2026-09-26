import requests
import pandas as pd


BASE_URL = "https://api.coingecko.com/api/v3"


COIN_IDS = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
    "LINK-USD": "chainlink",
    "ADA-USD": "cardano",
    "SUI-USD": "sui",
    "AVAX-USD": "avalanche-2",
    "TRX-USD": "tron",
    "HYPE-USD": "hyperliquid",
}


def download_crypto_info(tickers):
    """
    Download current cryptocurrency market information.
    """

    coin_ids = [
        COIN_IDS[ticker]
        for ticker in tickers
        if ticker in COIN_IDS
    ]

    if not coin_ids:
        raise ValueError(
            "No supported cryptocurrencies were selected."
        )

    params = {
        "vs_currency": "usd",
        "ids": ",".join(coin_ids),
        "order": "market_cap_desc",
        "per_page": len(coin_ids),
        "page": 1,
        "sparkline": "false",
    }

    response = requests.get(
        f"{BASE_URL}/coins/markets",
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise ValueError(
            f"CoinGecko API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    if not data:
        raise ValueError(
            "CoinGecko returned no data."
        )

    records = []

    for coin in data:

        records.append({
            "Name": coin.get("name"),
            "Symbol": coin.get(
                "symbol",
                ""
            ).upper(),

            "Rank": coin.get(
                "market_cap_rank"
            ),

            "Price": coin.get(
                "current_price"
            ),

            "Market Cap": coin.get(
                "market_cap"
            ),

            "24h Volume": coin.get(
                "total_volume"
            ),

            "24h Change": coin.get(
                "price_change_percentage_24h"
            ),

            "Circulating Supply": coin.get(
                "circulating_supply"
            ),

            "Total Supply": coin.get(
                "total_supply"
            ),

            "Max Supply": coin.get(
                "max_supply"
            ),

            "ATH": coin.get(
                "ath"
            ),

            "ATH Change %": coin.get(
                "ath_change_percentage"
            ),

            "ATL": coin.get(
                "atl"
            ),

            "ATL Change %": coin.get(
                "atl_change_percentage"
            ),

            "Last Updated": coin.get(
                "last_updated"
            ),

            "Image": coin.get(
                "image"
            ),

            "ID": coin.get(
                "id"
            ),
        })

    return pd.DataFrame(records)


def download_top_coins(limit=100):
    """
    Download the top cryptocurrencies
    ranked by market capitalization.
    """

    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": "false",
    }

    response = requests.get(
        f"{BASE_URL}/coins/markets",
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise ValueError(
            f"CoinGecko API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    if not data:
        raise ValueError(
            "CoinGecko returned no data."
        )

    records = []

    for coin in data:

        records.append({
            "Rank": coin.get(
                "market_cap_rank"
            ),

            "Name": coin.get(
                "name"
            ),

            "Symbol": coin.get(
                "symbol",
                ""
            ).upper(),

            "CoinGecko ID": coin.get(
                "id"
            ),

            "Price": coin.get(
                "current_price"
            ),

            "Market Cap": coin.get(
                "market_cap"
            ),

            "24h Volume": coin.get(
                "total_volume"
            ),

            "24h Change": coin.get(
                "price_change_percentage_24h"
            ),

            "Circulating Supply": coin.get(
                "circulating_supply"
            ),

            "Total Supply": coin.get(
                "total_supply"
            ),

            "Max Supply": coin.get(
                "max_supply"
            ),

            "ATH": coin.get(
                "ath"
            ),

            "ATH Change %": coin.get(
                "ath_change_percentage"
            ),

            "ATL": coin.get(
                "atl"
            ),

            "ATL Change %": coin.get(
                "atl_change_percentage"
            ),

            "Image": coin.get(
                "image"
            ),

            "Last Updated": coin.get(
                "last_updated"
            ),
        })

    return pd.DataFrame(records)


def download_market_chart(
    ticker,
    days=365
):
    """
    Download historical price data
    for one cryptocurrency.
    """

    if ticker not in COIN_IDS:
        raise ValueError(
            f"Unsupported cryptocurrency: {ticker}"
        )

    coin_id = COIN_IDS[ticker]

    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily",
    }

    response = requests.get(
        f"{BASE_URL}/coins/{coin_id}/market_chart",
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise ValueError(
            f"CoinGecko API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    prices = data.get(
        "prices",
        []
    )

    if not prices:
        raise ValueError(
            "CoinGecko returned no historical prices."
        )

    df = pd.DataFrame(
        prices,
        columns=[
            "timestamp",
            "price"
        ]
    )

    df["date"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )

    df = df[
        [
            "date",
            "price"
        ]
    ]

    df = df.drop_duplicates(
        subset="date"
    )

    df = df.sort_values(
        "date"
    )

    df = df.set_index(
        "date"
    )

    return df


def download_coin_details(ticker):
    """
    Download detailed information about one cryptocurrency.
    """

    if ticker not in COIN_IDS:
        raise ValueError(
            f"Unsupported cryptocurrency: {ticker}"
        )

    coin_id = COIN_IDS[ticker]

    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "false",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }

    response = requests.get(
        f"{BASE_URL}/coins/{coin_id}",
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise ValueError(
            f"CoinGecko API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    coin = response.json()

    links = coin.get(
        "links",
        {}
    )

    homepage = links.get(
        "homepage",
        []
    )

    whitepaper = links.get(
        "whitepaper"
    )

    blockchain_sites = links.get(
        "blockchain_site",
        []
    )

    return {
        "id": coin.get(
            "id"
        ),

        "name": coin.get(
            "name"
        ),

        "symbol": coin.get(
            "symbol",
            ""
        ).upper(),

        "description": coin.get(
            "description",
            {}
        ).get("en"),

        "genesis_date": coin.get(
            "genesis_date"
        ),

        "categories": coin.get(
            "categories",
            []
        ),

        "hashing_algorithm": coin.get(
            "hashing_algorithm"
        ),

        "block_time": coin.get(
            "block_time_in_minutes"
        ),

        "platforms": coin.get(
            "platforms",
            {}
        ),

        "contract_address": coin.get(
            "contract_address"
        ),

        "homepage": (
            homepage[0]
            if homepage
            else None
        ),

        "whitepaper": whitepaper,

        "blockchain_explorer": (
            blockchain_sites[0]
            if blockchain_sites
            else None
        ),
    }

def download_market_chart_by_id(coin_id, days=365):
    """
    Download historical market data using a CoinGecko coin ID.
    """

    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily",
    }

    response = requests.get(
        f"{BASE_URL}/coins/{coin_id}/market_chart",
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    prices = pd.DataFrame(
        data.get("prices", []),
        columns=["timestamp", "price"]
    )

    if prices.empty:
        raise ValueError(
            f"No historical price data found for {coin_id}."
        )

    prices["date"] = pd.to_datetime(
        prices["timestamp"],
        unit="ms"
    )

    prices = prices.set_index("date")

    return prices["price"]


def download_coin_details_by_id(coin_id):
    """
    Download detailed information for a CoinGecko coin ID.
    """

    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "true",
        "developer_data": "true",
        "sparkline": "false",
    }

    response = requests.get(
        f"{BASE_URL}/coins/{coin_id}",
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()