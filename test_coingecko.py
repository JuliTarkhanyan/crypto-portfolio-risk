from src.coingecko import (
    download_crypto_info,
    download_market_chart,
)


tickers = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "LINK-USD",
]


print("\nCURRENT MARKET DATA")
print("=" * 60)

info = download_crypto_info(tickers)

print(
    info.to_string(index=False)
)


print("\n\nBTC HISTORICAL DATA")
print("=" * 60)

btc = download_market_chart(
    "BTC-USD",
    days=365
)

print(btc.head())

print("\n...")

print(btc.tail())

print("\nNumber of observations:")
print(len(btc))