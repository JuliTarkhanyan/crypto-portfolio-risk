# ============================================================
# SRC/DATA.PY
# ============================================================

import json
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://api.coingecko.com/api/v3"
DEFILLAMA_BASE_URL = "https://api.llama.fi"

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PERSISTENT CACHE FILES
# ============================================================

TOP_COINS_CACHE = (
    CACHE_DIR / "top_coins.csv"
)

HISTORICAL_PRICES_CACHE = (
    CACHE_DIR / "historical_prices.parquet"
)

HISTORICAL_PRICES_CSV_CACHE = (
    CACHE_DIR / "historical_prices.csv"
)

COIN_DETAILS_CACHE_DIR = (
    CACHE_DIR / "coin_details"
)

DEFILLAMA_CACHE_DIR = (
    CACHE_DIR / "defillama"
)

DEFILLAMA_PROTOCOLS_CACHE = (
    DEFILLAMA_CACHE_DIR / "protocols.json"
)


COIN_DETAILS_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEFILLAMA_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CACHE POLICY
# ============================================================

# Cache never expires automatically.
#
# API is called only when:
#
#   1. cache does not exist
#   2. requested coin is missing from cache
#   3. force_refresh=True
#
# If an API refresh fails but old cached data exists,
# the old cached data is returned.

CACHE_NEVER_EXPIRES = True


# ============================================================
# FIXED COINS
# ============================================================

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


# ============================================================
# CACHE HELPERS
# ============================================================

def _cache_exists(path: Path) -> bool:
    """
    Return True if a cache file exists and is non-empty.
    """

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


def _read_json(path: Path):
    """
    Safely read a JSON cache.
    """

    if not _cache_exists(path):
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return None


def _write_json(
    path: Path,
    data,
):
    """
    Safely write JSON cache.
    """

    try:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return True

    except Exception:

        return False


# ============================================================
# COINGECKO REQUEST
# ============================================================

def _coingecko_get(
    endpoint,
    params=None,
    retries=3,
):
    """
    Make a CoinGecko API request.

    Handles:
        200 -> return data
        429 -> retry with exponential backoff
        other errors -> retry

    Returns:
        dict/list on success
        None on failure
    """

    url = (
        f"{BASE_URL}/{endpoint}"
    )

    for attempt in range(retries):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=30,
                headers={
                    "Accept": "application/json",
                    "User-Agent": (
                        "crypto-research-dashboard/1.0"
                    ),
                },
            )

            if response.status_code == 200:

                return response.json()

            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if response.status_code == 429:

                if attempt < retries - 1:

                    # 2, 4, 8 seconds
                    wait_seconds = (
                        2 ** (attempt + 1)
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue

                return None

            # ------------------------------------------------
            # OTHER HTTP ERROR
            # ------------------------------------------------

            response.raise_for_status()

        except Exception:

            if attempt < retries - 1:

                time.sleep(
                    2 ** attempt
                )

    return None


# ============================================================
# DEFI LLAMA REQUEST
# ============================================================

def _defillama_get(
    endpoint,
    retries=3,
):
    """
    Make a DeFiLlama API request.
    """

    url = (
        f"{DEFILLAMA_BASE_URL}/{endpoint}"
    )

    for attempt in range(retries):

        try:

            response = requests.get(
                url,
                timeout=30,
                headers={
                    "Accept": "application/json",
                    "User-Agent": (
                        "crypto-research-dashboard/1.0"
                    ),
                },
            )

            if response.status_code == 200:

                return response.json()

            if response.status_code == 429:

                if attempt < retries - 1:

                    time.sleep(
                        2 ** (attempt + 1)
                    )

                    continue

                return None

            response.raise_for_status()

        except Exception:

            if attempt < retries - 1:

                time.sleep(
                    2 ** attempt
                )

    return None


# ============================================================
# YAHOO FINANCE
# ============================================================

def download_prices(
    tickers,
    start_date,
    end_date,
):
    """
    Download historical prices from Yahoo Finance.

    Used by the fixed-coin analysis.
    """

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:

        raise ValueError(
            "No market data was returned."
        )

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        if "Close" in data.columns.levels[0]:

            prices = data["Close"]

        else:

            prices = data.xs(
                "Close",
                axis=1,
                level=0,
            )

    else:

        if "Close" in data.columns:

            prices = data[["Close"]]

        else:

            prices = data

    prices = (
        prices
        .sort_index()
        .dropna(
            axis=1,
            how="all",
        )
        .ffill()
    )

    return prices


# ============================================================
# TOP COINS
# ============================================================

def download_top_coins(
    limit=100,
    force_refresh=False,
):
    """
    Download Top cryptocurrencies from CoinGecko.

    Persistent cache:

        data/cache/top_coins.csv

    Normal behavior:
        use cache

    force_refresh=True:
        request fresh Top Coin metadata

    If refresh fails:
        return old cache
    """

    # ========================================================
    # USE CACHE
    # ========================================================

    if (
        not force_refresh
        and _cache_exists(
            TOP_COINS_CACHE
        )
    ):

        try:

            cached = pd.read_csv(
                TOP_COINS_CACHE
            )

            if not cached.empty:

                return cached.head(
                    limit
                )

        except Exception:

            pass

    # ========================================================
    # API REQUEST
    # ========================================================

    data = _coingecko_get(
        "coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": (
                "24h,7d,30d"
            ),
        },
    )

    # ========================================================
    # API FAILED
    # ========================================================

    if not data:

        if _cache_exists(
            TOP_COINS_CACHE
        ):

            try:

                cached = pd.read_csv(
                    TOP_COINS_CACHE
                )

                if not cached.empty:

                    return cached.head(
                        limit
                    )

            except Exception:

                pass

        raise ValueError(
            "Could not load Top cryptocurrencies "
            "from CoinGecko and no cached data exists."
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        data
    )

    if df.empty:

        raise ValueError(
            "CoinGecko returned an empty Top Coins response."
        )

    # ========================================================
    # ADD YAHOO TICKER
    # ========================================================

    if "symbol" in df.columns:

        df["Ticker"] = (
            df["symbol"]
            .astype(str)
            .str.upper()
            + "-USD"
        )

    # ========================================================
    # SAVE CACHE
    # ========================================================

    try:

        df.to_csv(
            TOP_COINS_CACHE,
            index=False,
        )

    except Exception:

        pass

    return df.head(
        limit
    )


# ============================================================
# HISTORICAL CACHE LOAD
# ============================================================

def _load_historical_cache():
    """
    Load persistent historical-price cache.

    Priority:

        1. Parquet
        2. CSV

    Returns:
        DataFrame
    """

    # ========================================================
    # PARQUET
    # ========================================================

    if _cache_exists(
        HISTORICAL_PRICES_CACHE
    ):

        try:

            cached = pd.read_parquet(
                HISTORICAL_PRICES_CACHE
            )

            if cached is not None:

                cached.index = pd.to_datetime(
                    cached.index,
                    errors="coerce",
                )

                cached = cached[
                    ~cached.index.isna()
                ]

                cached = (
                    cached
                    .sort_index()
                )

                return cached

        except Exception:

            pass

    # ========================================================
    # CSV FALLBACK
    # ========================================================

    if _cache_exists(
        HISTORICAL_PRICES_CSV_CACHE
    ):

        try:

            cached = pd.read_csv(
                HISTORICAL_PRICES_CSV_CACHE,
                index_col=0,
                parse_dates=True,
            )

            cached.index = pd.to_datetime(
                cached.index,
                errors="coerce",
            )

            cached = cached[
                ~cached.index.isna()
            ]

            return (
                cached
                .sort_index()
            )

        except Exception:

            pass

    return pd.DataFrame()


# ============================================================
# HISTORICAL CACHE SAVE
# ============================================================

def _save_historical_cache(
    prices,
):
    """
    Save historical prices to both:

        historical_prices.parquet
        historical_prices.csv

    Parquet is preferred.
    CSV is maintained as backup.
    """

    if (
        prices is None
        or prices.empty
    ):

        return

    prices = (
        prices
        .copy()
        .sort_index()
    )

    # Normalize index
    prices.index = pd.to_datetime(
        prices.index,
        errors="coerce",
    )

    prices = prices[
        ~prices.index.isna()
    ]

    # ========================================================
    # PARQUET
    # ========================================================

    try:

        prices.to_parquet(
            HISTORICAL_PRICES_CACHE
        )

    except Exception:

        pass

    # ========================================================
    # CSV BACKUP
    # ========================================================

    try:

        prices.to_csv(
            HISTORICAL_PRICES_CSV_CACHE
        )

    except Exception:

        pass


# ============================================================
# CONVERT COINGECKO MARKET CHART
# ============================================================

def _convert_market_chart_to_dataframe(
    data,
    coin_id,
):
    """
    Convert CoinGecko market_chart response
    into a one-column DataFrame.
    """

    if (
        not data
        or "prices" not in data
    ):

        return pd.DataFrame()

    raw_prices = data.get(
        "prices",
        [],
    )

    if not raw_prices:

        return pd.DataFrame()

    prices = pd.DataFrame(
        raw_prices,
        columns=[
            "timestamp",
            "price",
        ],
    )

    if prices.empty:

        return pd.DataFrame()

    prices["Date"] = pd.to_datetime(
        prices["timestamp"],
        unit="ms",
        errors="coerce",
    ).dt.normalize()

    prices["price"] = pd.to_numeric(
        prices["price"],
        errors="coerce",
    )

    prices = prices[
        [
            "Date",
            "price",
        ]
    ]

    prices = prices.dropna(
        subset=[
            "Date",
            "price",
        ]
    )

    if prices.empty:

        return pd.DataFrame()

    prices = (
        prices
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .set_index(
            "Date"
        )
        .rename(
            columns={
                "price": coin_id
            }
        )
    )

    return prices.sort_index()


# ============================================================
# SINGLE COIN MARKET CHART
# ============================================================

def download_market_chart_by_id(
    coin_id,
    days=365,
    force_refresh=False,
):
    """
    Download historical CoinGecko prices for one coin.

    Persistent global cache:

        data/cache/historical_prices.parquet

    Normal behavior:
        if coin exists in cache -> NO API REQUEST

    Missing coin:
        API request

    force_refresh=True:
        API request for THIS COIN ONLY

    This is important because Single Crypto Research
    should never need to refresh all 100 cryptocurrencies.
    """

    coin_id = str(
        coin_id
    ).strip()

    if not coin_id:

        return pd.DataFrame()

    # ========================================================
    # LOAD CURRENT CACHE
    # ========================================================

    cached_prices = (
        _load_historical_cache()
    )

    # ========================================================
    # USE CACHE
    # ========================================================

    if (
        not force_refresh
        and not cached_prices.empty
        and coin_id in cached_prices.columns
    ):

        result = (
            cached_prices[
                [coin_id]
            ]
            .copy()
            .sort_index()
        )

        return result

    # ========================================================
    # API REQUEST — THIS COIN ONLY
    # ========================================================

    data = _coingecko_get(
        f"coins/{coin_id}/market_chart",
        params={
            "vs_currency": "usd",
            "days": days,
            "interval": "daily",
        },
    )

    # ========================================================
    # API FAILED
    # ========================================================

    if (
        not data
        or "prices" not in data
    ):

        # Always use old cache if available.
        if (
            not cached_prices.empty
            and coin_id in cached_prices.columns
        ):

            return (
                cached_prices[
                    [coin_id]
                ]
                .copy()
                .sort_index()
            )

        return pd.DataFrame()

    # ========================================================
    # CONVERT
    # ========================================================

    prices = (
        _convert_market_chart_to_dataframe(
            data,
            coin_id,
        )
    )

    if prices.empty:

        # Old cache fallback.
        if (
            not cached_prices.empty
            and coin_id in cached_prices.columns
        ):

            return (
                cached_prices[
                    [coin_id]
                ]
                .copy()
                .sort_index()
            )

        return pd.DataFrame()

    # ========================================================
    # MERGE INTO GLOBAL CACHE
    # ========================================================

    if cached_prices.empty:

        cached_prices = prices.copy()

    else:

        cached_prices = (
            cached_prices.copy()
        )

        # Replace only this coin.
        cached_prices[coin_id] = (
            prices[coin_id]
        )

    cached_prices = (
        cached_prices
        .sort_index()
    )

    # ========================================================
    # SAVE
    # ========================================================

    _save_historical_cache(
        cached_prices
    )

    return prices.sort_index()


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def download_market_chart(
    coin_id,
    days=365,
    force_refresh=False,
):
    """
    Backward-compatible wrapper.
    """

    return download_market_chart_by_id(
        coin_id=coin_id,
        days=days,
        force_refresh=force_refresh,
    )


# ============================================================
# MULTI-COIN HISTORICAL PRICES
# ============================================================

def download_historical_prices_by_ids(
    coin_ids,
    days=365,
    force_refresh=False,
):
    """
    Download historical prices for multiple CoinGecko IDs.

    Behavior:

        force_refresh=False

            Cached coin -> no API request
            Missing coin -> API request

        force_refresh=True

            Every requested coin -> API request

    Importantly, this function does NOT download
    coins outside the requested list.
    """

    # ========================================================
    # CLEAN IDS
    # ========================================================

    coin_ids = list(
        dict.fromkeys(
            [
                str(x).strip()
                for x in coin_ids
                if x
            ]
        )
    )

    if not coin_ids:

        return pd.DataFrame()

    # ========================================================
    # LOAD CACHE
    # ========================================================

    cached_prices = (
        _load_historical_cache()
    )

    # ========================================================
    # PROCESS REQUESTED COINS ONLY
    # ========================================================

    for coin_id in coin_ids:

        already_cached = (
            not cached_prices.empty
            and coin_id in cached_prices.columns
        )

        # ----------------------------------------------------
        # USE CACHE
        # ----------------------------------------------------

        if (
            already_cached
            and not force_refresh
        ):

            continue

        # ----------------------------------------------------
        # DOWNLOAD ONE COIN
        # ----------------------------------------------------

        prices = (
            download_market_chart_by_id(
                coin_id=coin_id,
                days=days,
                force_refresh=force_refresh,
            )
        )

        # ----------------------------------------------------
        # MERGE SUCCESSFUL DOWNLOAD
        # ----------------------------------------------------

        if not prices.empty:

            if cached_prices.empty:

                cached_prices = (
                    prices.copy()
                )

            else:

                cached_prices = (
                    cached_prices.copy()
                )

                cached_prices[coin_id] = (
                    prices[coin_id]
                )

            cached_prices = (
                cached_prices
                .sort_index()
            )

            # Save immediately.
            _save_historical_cache(
                cached_prices
            )

        # ----------------------------------------------------
        # RATE-LIMIT PROTECTION
        # ----------------------------------------------------

        # Only sleep when making API calls.
        #
        # 0.25 seconds keeps bulk downloads
        # slightly more conservative.
        if (
            not already_cached
            or force_refresh
        ):

            time.sleep(
                0.25
            )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    if cached_prices.empty:

        return pd.DataFrame()

    available = [
        coin
        for coin in coin_ids
        if coin in cached_prices.columns
    ]

    if not available:

        return pd.DataFrame()

    result = (
        cached_prices[
            available
        ]
        .copy()
        .sort_index()
    )

    return result


# ============================================================
# TOP 100 HISTORICAL PRICES
# ============================================================

def download_top_100_historical_prices(
    days=365,
    force_refresh=False,
    coin_ids=None,
):
    """
    Download historical prices for the CoinGecko Top 100.

    Parameters
    ----------
    days:
        Historical period.

    force_refresh:
        If False:
            use cache wherever possible.

        If True:
            refresh ONLY the requested coin_ids.

            If coin_ids is None:
            refresh the complete Top 100.

    coin_ids:
        Optional list of specific CoinGecko IDs.

        This is especially useful for Single Crypto
        Research because it can refresh one coin without
        triggering 100 CoinGecko API requests.

    Examples
    --------

    Main portfolio:

        download_top_100_historical_prices(
            days=365,
            force_refresh=False,
        )

    Full Top 100 refresh:

        download_top_100_historical_prices(
            days=365,
            force_refresh=True,
        )

    Single coin refresh:

        download_top_100_historical_prices(
            days=365,
            force_refresh=True,
            coin_ids=["ethereum"],
        )
    """

    # ========================================================
    # DETERMINE REQUESTED COINS
    # ========================================================

    if coin_ids is not None:

        requested_ids = list(
            dict.fromkeys(
                [
                    str(x).strip()
                    for x in coin_ids
                    if x
                ]
            )
        )

        if not requested_ids:

            return pd.DataFrame()

    else:

        # Get current Top 100 metadata.
        top_coins = (
            download_top_coins(
                limit=100,
                force_refresh=force_refresh,
            )
        )

        if top_coins.empty:

            return pd.DataFrame()

        if "id" not in top_coins.columns:

            return pd.DataFrame()

        requested_ids = (
            top_coins["id"]
            .dropna()
            .astype(str)
            .tolist()
        )

    # ========================================================
    # LOAD REQUESTED HISTORICAL DATA
    # ========================================================

    return (
        download_historical_prices_by_ids(
            coin_ids=requested_ids,
            days=days,
            force_refresh=force_refresh,
        )
    )


# ============================================================
# COIN DETAILS CACHE PATH
# ============================================================

def _coin_details_cache_path(
    coin_id,
):
    """
    Return persistent cache path for a coin.
    """

    safe_coin_id = (
        str(coin_id)
        .strip()
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
    )

    return (
        COIN_DETAILS_CACHE_DIR
        / f"{safe_coin_id}.json"
    )


# ============================================================
# COIN DETAILS
# ============================================================

def download_coin_details_by_id(
    coin_id,
    force_refresh=False,
):
    """
    Download detailed information for one CoinGecko coin.

    Persistent cache:

        data/cache/coin_details/<coin_id>.json

    Normal:
        use cache

    force_refresh=True:
        refresh THIS coin only

    If API refresh fails:
        use old cache
    """

    coin_id = str(
        coin_id
    ).strip()

    if not coin_id:

        return None

    cache_path = (
        _coin_details_cache_path(
            coin_id
        )
    )

    # ========================================================
    # CACHE
    # ========================================================

    if (
        not force_refresh
        and _cache_exists(
            cache_path
        )
    ):

        cached = _read_json(
            cache_path
        )

        if cached is not None:

            return cached

    # ========================================================
    # API
    # ========================================================

    data = _coingecko_get(
        f"coins/{coin_id}",
        params={
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
        },
    )

    # ========================================================
    # API FAILED
    # ========================================================

    if data is None:

        cached = _read_json(
            cache_path
        )

        if cached is not None:

            return cached

        raise ValueError(
            f"Could not load CoinGecko details "
            f"for {coin_id}."
        )

    # ========================================================
    # SAVE
    # ========================================================

    _write_json(
        cache_path,
        data,
    )

    return data


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def download_coin_details(
    coin_id,
    force_refresh=False,
):
    """
    Backward-compatible wrapper.
    """

    return download_coin_details_by_id(
        coin_id=coin_id,
        force_refresh=force_refresh,
    )


# ============================================================
# FIXED COIN INFO
# ============================================================

def download_crypto_info(
    ticker,
    force_refresh=False,
):
    """
    Get CoinGecko information using COIN_IDS.
    """

    if ticker not in COIN_IDS:

        raise ValueError(
            f"Unknown ticker: {ticker}"
        )

    coin_id = COIN_IDS[
        ticker
    ]

    return download_coin_details_by_id(
        coin_id=coin_id,
        force_refresh=force_refresh,
    )


# ============================================================
# DEFILLAMA PROTOCOL LIST
# ============================================================

def download_defillama_protocols(
    force_refresh=False,
):
    """
    Download the DeFiLlama protocol list.

    Persistent cache:

        data/cache/defillama/protocols.json
    """

    # ========================================================
    # CACHE
    # ========================================================

    if (
        not force_refresh
        and _cache_exists(
            DEFILLAMA_PROTOCOLS_CACHE
        )
    ):

        cached = _read_json(
            DEFILLAMA_PROTOCOLS_CACHE
        )

        if cached is not None:

            return cached

    # ========================================================
    # API
    # ========================================================

    data = _defillama_get(
        "protocols"
    )

    # ========================================================
    # FALLBACK
    # ========================================================

    if data is None:

        cached = _read_json(
            DEFILLAMA_PROTOCOLS_CACHE
        )

        if cached is not None:

            return cached

        return []

    # ========================================================
    # SAVE
    # ========================================================

    _write_json(
        DEFILLAMA_PROTOCOLS_CACHE,
        data,
    )

    return data


# ============================================================
# DEFILLAMA SINGLE PROTOCOL CACHE PATH
# ============================================================

def _defillama_protocol_cache_path(
    slug,
):
    """
    Persistent cache path for a DeFiLlama protocol.
    """

    safe_slug = (
        str(slug)
        .strip()
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
    )

    return (
        DEFILLAMA_CACHE_DIR
        / f"protocol_{safe_slug}.json"
    )


# ============================================================
# DEFILLAMA SINGLE PROTOCOL
# ============================================================

def download_defillama_protocol(
    slug,
    force_refresh=False,
):
    """
    Download details for one DeFiLlama protocol.

    Persistent cache:

        data/cache/defillama/protocol_<slug>.json
    """

    if not slug:

        return None

    cache_path = (
        _defillama_protocol_cache_path(
            slug
        )
    )

    # ========================================================
    # CACHE
    # ========================================================

    if (
        not force_refresh
        and _cache_exists(
            cache_path
        )
    ):

        cached = _read_json(
            cache_path
        )

        if cached is not None:

            return cached

    # ========================================================
    # API
    # ========================================================

    data = _defillama_get(
        f"protocol/{slug}"
    )

    # ========================================================
    # FALLBACK
    # ========================================================

    if data is None:

        cached = _read_json(
            cache_path
        )

        if cached is not None:

            return cached

        return None

    # ========================================================
    # SAVE
    # ========================================================

    _write_json(
        cache_path,
        data,
    )

    return data


# ============================================================
# CACHE MANAGEMENT
# ============================================================

def clear_price_cache():
    """
    Delete all historical price caches.
    """

    try:

        if HISTORICAL_PRICES_CACHE.exists():

            HISTORICAL_PRICES_CACHE.unlink()

    except Exception:

        pass

    try:

        if (
            HISTORICAL_PRICES_CSV_CACHE.exists()
        ):

            HISTORICAL_PRICES_CSV_CACHE.unlink()

    except Exception:

        pass


def clear_top_coins_cache():
    """
    Delete Top Coins metadata cache.
    """

    try:

        if TOP_COINS_CACHE.exists():

            TOP_COINS_CACHE.unlink()

    except Exception:

        pass


def clear_coin_details_cache():
    """
    Delete all CoinGecko details caches.
    """

    if not COIN_DETAILS_CACHE_DIR.exists():

        return

    for file in (
        COIN_DETAILS_CACHE_DIR.glob(
            "*.json"
        )
    ):

        try:

            file.unlink()

        except Exception:

            pass


def clear_defillama_cache():
    """
    Delete all DeFiLlama caches.
    """

    if not DEFILLAMA_CACHE_DIR.exists():

        return

    for file in (
        DEFILLAMA_CACHE_DIR.glob(
            "*.json"
        )
    ):

        try:

            file.unlink()

        except Exception:

            pass


def clear_all_cache():
    """
    Delete ALL persistent API caches.
    """

    clear_price_cache()
    clear_top_coins_cache()
    clear_coin_details_cache()
    clear_defillama_cache()