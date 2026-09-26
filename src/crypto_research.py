from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote_plus, urlparse

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# LOCAL CACHE
# ============================================================

RESEARCH_DIR = Path("data") / "crypto_research"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "CryptoPortfolioRiskAnalysis/1.0"
REQUEST_TIMEOUT = 15


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_json_value(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429:
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _get_text(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429:
            return None
        response.raise_for_status()
        return response.text
    except Exception:
        return None


def _load_cache(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _save_cache(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(
                _clean_json_value(payload),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


# ============================================================
# YAHOO FINANCE MARKET DATA
# ============================================================

def _history_from_yahoo(yahoo_symbol: str, period: str = "1y") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(yahoo_symbol)
        history = ticker.history(
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
    except Exception:
        return pd.DataFrame()

    if history is None or history.empty:
        return pd.DataFrame()

    if isinstance(history.columns, pd.MultiIndex):
        try:
            history.columns = history.columns.get_level_values(0)
        except Exception:
            pass

    history.index = pd.to_datetime(history.index)

    if getattr(history.index, "tz", None) is not None:
        history.index = history.index.tz_localize(None)

    columns = [c for c in ["Close", "Volume"] if c in history.columns]
    if not columns:
        return pd.DataFrame()

    return history[columns].dropna(how="all")


def get_yahoo_market_data(symbol: str) -> Dict[str, Any]:
    symbol = str(symbol or "").upper().strip()

    if not symbol:
        return {
            "history": {},
            "current_price": None,
            "volume_latest": None,
            "annualized_volatility": None,
            "maximum_drawdown": None,
            "data_source": "Yahoo Finance",
        }

    yahoo_symbol = symbol if symbol.endswith("-USD") else f"{symbol}-USD"
    history = _history_from_yahoo(yahoo_symbol, period="1y")

    if history.empty or "Close" not in history.columns:
        return {
            "history": {},
            "current_price": None,
            "volume_latest": None,
            "annualized_volatility": None,
            "maximum_drawdown": None,
            "data_source": "Yahoo Finance",
        }

    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    volume = pd.to_numeric(
        history.get("Volume", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()

    daily_returns = close.pct_change().dropna()

    volatility = None
    if len(daily_returns) > 1:
        volatility = _safe_float(daily_returns.std(ddof=1) * np.sqrt(365))

    max_drawdown = None
    if len(close) > 1:
        drawdown = close / close.cummax() - 1
        max_drawdown = _safe_float(drawdown.min())

    history_dict = {
        pd.Timestamp(index).strftime("%Y-%m-%d"): _safe_float(value)
        for index, value in close.items()
        if _safe_float(value) is not None
    }

    return {
        "history": history_dict,
        "current_price": _safe_float(close.iloc[-1]) if not close.empty else None,
        "volume_latest": _safe_float(volume.iloc[-1]) if not volume.empty else None,
        "annualized_volatility": volatility,
        "maximum_drawdown": max_drawdown,
        "data_source": "Yahoo Finance",
        "yahoo_symbol": yahoo_symbol,
        "observations": int(len(close)),
    }


# ============================================================
# COINGECKO PROJECT + TOKENOMICS
# ============================================================

COINGECKO_URL = "https://api.coingecko.com/api/v3"


def _extract_links(links: Dict[str, Any]) -> Dict[str, Any]:
    websites = [x for x in links.get("homepage", []) if x]
    repos = []
    for repo_group in ["repos_url"]:
        group = links.get(repo_group, {}) or {}
        for key in ["github", "bitbucket"]:
            values = group.get(key, []) or []
            repos.extend([x for x in values if x])

    return {
        "website": websites[0] if websites else None,
        "websites": websites,
        "github": repos[0] if repos else None,
        "repositories": repos,
        "whitepaper": (links.get("whitepaper") or None),
        "twitter_screen_name": links.get("twitter_screen_name") or None,
        "subreddit_url": links.get("subreddit_url") or None,
        "telegram_channel_identifier": links.get("telegram_channel_identifier") or None,
        "discord": links.get("discord") or None,
    }


def fetch_coingecko_coin_details(coin_id: str) -> Dict[str, Any]:
    data = _get_json(
        f"{COINGECKO_URL}/coins/{quote_plus(str(coin_id))}",
        params={
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        },
    )

    if not isinstance(data, dict):
        return {}

    market = data.get("market_data", {}) or {}
    links = _extract_links(data.get("links", {}) or {})
    description_raw = (data.get("description", {}) or {}).get("en", "")
    description = re.sub(r"<[^>]+>", " ", description_raw or "")
    description = html.unescape(re.sub(r"\s+", " ", description)).strip()

    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "symbol": str(data.get("symbol") or "").upper(),
        "description": description,
        "categories": data.get("categories", []) or [],
        "genesis_date": data.get("genesis_date"),
        "country_origin": data.get("country_origin"),
        "hashing_algorithm": data.get("hashing_algorithm"),
        "asset_platform_id": data.get("asset_platform_id"),
        "platforms": data.get("platforms", {}) or {},
        "contract_address": data.get("contract_address"),
        "links": links,
        "market": {
            "current_price": _safe_float((market.get("current_price") or {}).get("usd")),
            "market_cap": _safe_float((market.get("market_cap") or {}).get("usd")),
            "fully_diluted_valuation": _safe_float((market.get("fully_diluted_valuation") or {}).get("usd")),
            "circulating_supply": _safe_float(market.get("circulating_supply")),
            "total_supply": _safe_float(market.get("total_supply")),
            "max_supply": _safe_float(market.get("max_supply")),
            "market_cap_rank": data.get("market_cap_rank"),
            "ath": _safe_float((market.get("ath") or {}).get("usd")),
            "atl": _safe_float((market.get("atl") or {}).get("usd")),
            "ath_change_percentage": _safe_float((market.get("ath_change_percentage") or {}).get("usd")),
            "atl_change_percentage": _safe_float((market.get("atl_change_percentage") or {}).get("usd")),
            "price_change_percentage_7d": _safe_float(market.get("price_change_percentage_7d_in_currency", {}).get("usd")),
            "price_change_percentage_30d": _safe_float(market.get("price_change_percentage_30d_in_currency", {}).get("usd")),
            "price_change_percentage_1y": _safe_float(market.get("price_change_percentage_1y_in_currency", {}).get("usd")),
        },
        "community": {
            "twitter_followers": _safe_float((data.get("community_data") or {}).get("twitter_followers")),
            "reddit_subscribers": _safe_float((data.get("community_data") or {}).get("reddit_subscribers")),
            "telegram_channel_user_count": _safe_float((data.get("community_data") or {}).get("telegram_channel_user_count")),
        },
        "developer": data.get("developer_data", {}) or {},
        "coingecko": {
            "last_updated": data.get("last_updated"),
            "sentiment_votes_up_percentage": _safe_float(data.get("sentiment_votes_up_percentage")),
            "sentiment_votes_down_percentage": _safe_float(data.get("sentiment_votes_down_percentage")),
        },
    }


# ============================================================
# DEFILLAMA
# ============================================================

DEFILLAMA_PROTOCOLS_URL = "https://api.llama.fi/protocols"
DEFILLAMA_EMISSIONS_URL = "https://api.llama.fi/emissions"


def fetch_defillama_protocols() -> list[Dict[str, Any]]:
    cache_path = RESEARCH_DIR / "_defillama_protocols.json"
    cached = _load_cache(cache_path)
    if cached and isinstance(cached.get("data"), list):
        cached_at = cached.get("cached_at")
        if cached_at:
            try:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(cached_at)
                if age.total_seconds() < 6 * 3600:
                    return cached["data"]
            except Exception:
                pass

    data = _get_json(DEFILLAMA_PROTOCOLS_URL)
    if isinstance(data, list):
        _save_cache(
            cache_path,
            {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "data": data,
            },
        )
        return data

    return cached.get("data", []) if cached else []


def find_defillama_protocol(coin_id: str) -> Dict[str, Any]:
    protocols = fetch_defillama_protocols()
    coin_id_lower = str(coin_id).lower()

    candidates = []
    for protocol in protocols:
        gecko_id = str(protocol.get("gecko_id") or "").lower()
        name = str(protocol.get("name") or "").lower()
        slug = str(protocol.get("slug") or protocol.get("id") or "").lower()

        if gecko_id == coin_id_lower:
            return protocol

        if coin_id_lower in {name, slug}:
            candidates.append(protocol)

    return candidates[0] if candidates else {}


def fetch_defillama_emission(coin_id: str) -> Dict[str, Any]:
    data = _get_json(DEFILLAMA_EMISSIONS_URL)
    if isinstance(data, list):
        for item in data:
            if str(item.get("gecko_id") or "").lower() == str(coin_id).lower():
                return item
    return {}


# ============================================================
# NEWS / EVENTS VIA GOOGLE NEWS RSS
# ============================================================

def _parse_google_news_rss(xml_text: Optional[str], query: str, limit: int = 8) -> list[Dict[str, Any]]:
    if not xml_text:
        return []

    items = re.findall(r"<item>(.*?)</item>", xml_text, flags=re.DOTALL | re.IGNORECASE)
    results = []

    for item in items[:limit]:
        title_match = re.search(r"<title>(.*?)</title>", item, flags=re.DOTALL | re.IGNORECASE)
        link_match = re.search(r"<link>(.*?)</link>", item, flags=re.DOTALL | re.IGNORECASE)
        date_match = re.search(r"<pubDate>(.*?)</pubDate>", item, flags=re.DOTALL | re.IGNORECASE)
        source_match = re.search(r"<source[^>]*>(.*?)</source>", item, flags=re.DOTALL | re.IGNORECASE)

        title = html.unescape(re.sub(r"\s+", " ", title_match.group(1))).strip() if title_match else ""
        link = html.unescape(link_match.group(1).strip()) if link_match else ""
        pub_date = html.unescape(re.sub(r"\s+", " ", date_match.group(1))).strip() if date_match else ""
        source = html.unescape(re.sub(r"\s+", " ", source_match.group(1))).strip() if source_match else ""

        if title:
            results.append(
                {
                    "title": title,
                    "source": source,
                    "published": pub_date,
                    "url": link,
                    "query": query,
                }
            )

    return results


def fetch_news_events(name: str, symbol: str) -> Dict[str, list[Dict[str, Any]]]:
    search_name = str(name or symbol).strip()
    queries = {
        "funding": f'"{search_name}" crypto funding investors venture',
        "events": f'"{search_name}" crypto partnership upgrade launch hack exploit regulation',
        "history": f'"{search_name}" crypto launch mainnet history',
        "unlocks": f'"{search_name}" token unlock vesting investors',
    }

    result = {}
    for key, query in queries.items():
        xml = _get_text(
            "https://news.google.com/rss/search",
            params={
                "q": query,
                "hl": "en-US",
                "gl": "US",
                "ceid": "US:en",
            },
        )
        result[key] = _parse_google_news_rss(xml, query=query, limit=8)
        time.sleep(0.2)

    return result


# ============================================================
# GITHUB
# ============================================================

def _github_repo_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None

    parsed = urlparse(url)
    if parsed.netloc.lower() != "github.com":
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"

    return None


def fetch_github_data(github_url: Optional[str]) -> Dict[str, Any]:
    repo = _github_repo_from_url(github_url)
    if not repo:
        return {}

    data = _get_json(f"https://api.github.com/repos/{repo}")
    if not isinstance(data, dict):
        return {
            "repository": github_url,
            "repository_name": repo,
            "status": "Repository identified; GitHub statistics unavailable from public request.",
        }

    return {
        "repository": data.get("html_url") or github_url,
        "repository_name": repo,
        "description": data.get("description"),
        "stars": data.get("stargazers_count"),
        "forks": data.get("forks_count"),
        "open_issues": data.get("open_issues_count"),
        "watchers": data.get("watchers_count"),
        "default_branch": data.get("default_branch"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "language": data.get("language"),
        "license": (data.get("license") or {}).get("spdx_id") if isinstance(data.get("license"), dict) else None,
        "topics": data.get("topics", []) or [],
    }


# ============================================================
# DERIVED METRICS
# ============================================================

def _trend_metrics(prices: pd.Series) -> Dict[str, Optional[float]]:
    prices = pd.Series(prices, dtype="float64").dropna().sort_index()
    if prices.empty:
        return {"7D": None, "30D": None, "90D": None, "180D": None, "1Y": None}

    latest = float(prices.iloc[-1])
    result: Dict[str, Optional[float]] = {}

    for label, days in {"7D": 7, "30D": 30, "90D": 90, "180D": 180, "1Y": 365}.items():
        if len(prices) > days:
            old_price = float(prices.iloc[-days - 1])
            result[label] = (latest / old_price - 1) if old_price else None
        else:
            result[label] = None

    return result


def _derived_risks(market: Dict[str, Any], tokenomics: Dict[str, Any], project: Dict[str, Any]) -> list[str]:
    risks: list[str] = []

    circulating = _safe_float(tokenomics.get("circulating_supply"))
    max_supply = _safe_float(tokenomics.get("max_supply"))
    fdv = _safe_float(tokenomics.get("fully_diluted_valuation"))
    market_cap = _safe_float(tokenomics.get("market_cap"))

    if circulating and max_supply and max_supply > 0:
        ratio = circulating / max_supply
        if ratio < 0.5:
            risks.append(
                f"Circulating supply is about {ratio:.1%} of stated maximum supply, so future supply expansion should be monitored."
            )

    if market_cap and fdv and market_cap > 0 and fdv > market_cap * 1.5:
        premium = fdv / market_cap - 1
        risks.append(
            f"Fully diluted valuation is about {premium:.0%} above current market capitalization, indicating material dilution sensitivity."
        )

    volatility = _safe_float(market.get("annualized_volatility"))
    if volatility is not None and volatility > 1.0:
        risks.append("The trailing 1-year annualized price volatility is above 100%.")

    max_drawdown = _safe_float(market.get("maximum_drawdown"))
    if max_drawdown is not None and max_drawdown < -0.70:
        risks.append("The trailing 1-year price history contains a drawdown greater than 70%.")

    categories = {str(x).lower() for x in project.get("categories", [])}
    if any("bridge" in x for x in categories):
        risks.append("Bridge-related assets can carry additional smart-contract and infrastructure risk.")
    if any("defi" in x for x in categories):
        risks.append("DeFi-related usage introduces application and smart-contract dependency.")

    if not risks:
        risks.append("No automatic high-priority risk flag was derived from the collected public metrics.")

    return risks


# ============================================================
# MAIN RESEARCH PIPELINE
# ============================================================

def refresh_coin_research(
    coin_id: str,
    name: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    coin_id = str(coin_id)
    name = str(name or coin_id)
    symbol = str(symbol or "").upper().strip()

    cache_path = RESEARCH_DIR / f"{coin_id}.json"

    yahoo_market = get_yahoo_market_data(symbol)
    coingecko = fetch_coingecko_coin_details(coin_id)
    defillama = find_defillama_protocol(coin_id)
    emissions = fetch_defillama_emission(coin_id)

    project_links = (coingecko.get("links") or {})
    github = fetch_github_data(project_links.get("github"))
    news = fetch_news_events(name, symbol)

    history_prices = pd.Series(
        yahoo_market.get("history", {}),
        dtype="float64",
    )
    if not history_prices.empty:
        history_prices.index = pd.to_datetime(history_prices.index)
        history_prices = history_prices.sort_index()

    trends = _trend_metrics(history_prices)

    cg_market = coingecko.get("market", {}) or {}

    market = {
        "current_price": yahoo_market.get("current_price") or cg_market.get("current_price"),
        "volume_latest": yahoo_market.get("volume_latest"),
        "annualized_volatility": yahoo_market.get("annualized_volatility"),
        "maximum_drawdown": yahoo_market.get("maximum_drawdown"),
        "market_cap": cg_market.get("market_cap"),
        "fully_diluted_valuation": cg_market.get("fully_diluted_valuation"),
        "market_cap_rank": coingecko.get("market", {}).get("market_cap_rank"),
        "price_change_percentage_7d": cg_market.get("price_change_percentage_7d"),
        "price_change_percentage_30d": cg_market.get("price_change_percentage_30d"),
        "price_change_percentage_1y": cg_market.get("price_change_percentage_1y"),
        "ath": cg_market.get("ath"),
        "atl": cg_market.get("atl"),
        "ath_change_percentage": cg_market.get("ath_change_percentage"),
        "atl_change_percentage": cg_market.get("atl_change_percentage"),
        "data_sources": ["Yahoo Finance", "CoinGecko"],
    }

    tokenomics = {
        "current_price": cg_market.get("current_price"),
        "market_cap": cg_market.get("market_cap"),
        "fully_diluted_valuation": cg_market.get("fully_diluted_valuation"),
        "circulating_supply": cg_market.get("circulating_supply"),
        "total_supply": cg_market.get("total_supply"),
        "max_supply": cg_market.get("max_supply"),
        "circulating_to_max_ratio": (
            cg_market.get("circulating_supply") / cg_market.get("max_supply")
            if cg_market.get("circulating_supply") is not None
            and cg_market.get("max_supply") not in (None, 0)
            else None
        ),
        "defillama_emissions": emissions or {},
        "data_source": "CoinGecko + DefiLlama when available",
    }

    project = {
        "name": coingecko.get("name") or name,
        "symbol": coingecko.get("symbol") or symbol,
        "description": coingecko.get("description"),
        "categories": coingecko.get("categories", []),
        "genesis_date": coingecko.get("genesis_date"),
        "country_origin": coingecko.get("country_origin"),
        "hashing_algorithm": coingecko.get("hashing_algorithm"),
        "asset_platform_id": coingecko.get("asset_platform_id"),
        "platforms": coingecko.get("platforms", {}),
        "contract_address": coingecko.get("contract_address", {}),
        "website": project_links.get("website"),
        "whitepaper": project_links.get("whitepaper"),
        "github": project_links.get("github"),
    }

    investors = {
        "status": "Derived from public news search; not a canonical cap table.",
        "items": news.get("funding", []),
    }

    funding = {
        "status": "Derived from public news search; verify against official disclosures.",
        "items": news.get("funding", []),
    }

    unlocks = {
        "defillama_emissions": emissions or {},
        "news": news.get("unlocks", []),
        "note": "No investor/team unlock figure is invented when a public source does not provide one.",
    }

    history = {
        "genesis_date": coingecko.get("genesis_date"),
        "items": news.get("history", []),
    }

    ecosystem = {
        "defillama": defillama or {},
        "tvl": _safe_float((defillama or {}).get("tvl")),
        "chains": (defillama or {}).get("chains", []) or [],
        "category": (defillama or {}).get("category"),
        "url": (defillama or {}).get("url"),
        "description": (defillama or {}).get("description"),
    }

    events = {
        "news": news.get("events", []),
        "note": "Events are collected from public news search results and should be verified before being treated as definitive project history.",
    }

    risks = _derived_risks(market, tokenomics, project)

    research = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "coin": {
                "id": coin_id,
                "name": project.get("name"),
                "symbol": project.get("symbol"),
            },
            "market": market,
            "project": project,
            "price_history": yahoo_market.get("history", {}),
            "price_trends": trends,
            "tokenomics": tokenomics,
            "investors": investors,
            "funding": funding,
            "unlocks": unlocks,
            "history": history,
            "ecosystem": ecosystem,
            "github": github,
            "events": events,
            "risks": risks,
            "community": coingecko.get("community", {}),
            "developer": coingecko.get("developer", {}),
            "sources": {
                "coingecko": f"https://www.coingecko.com/en/coins/{coin_id}",
                "yahoo_finance": yahoo_market.get("yahoo_symbol"),
                "defillama": (defillama or {}).get("url"),
                "github": project_links.get("github"),
            },
        },
    }

    _save_cache(cache_path, research)
    return research


def get_coin_research(
    coin_id: str,
    refresh: bool = False,
    name: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    path = RESEARCH_DIR / f"{coin_id}.json"

    if not refresh:
        cached = _load_cache(path)
        if cached and cached.get("data"):
            return cached

    return refresh_coin_research(
        coin_id,
        name=name,
        symbol=symbol,
    )


def calculate_price_trends(prices: pd.Series) -> Dict[str, Optional[float]]:
    return _trend_metrics(prices)
