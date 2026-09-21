from src.data import download_prices


tickers = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "LINK-USD"
]

prices = download_prices(
    tickers,
    "2025-01-01",
    "2026-01-01"
)

print(prices.head())
print()
print(prices.tail())
print()
print(prices.shape)