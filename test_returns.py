from src.data import download_prices
from src.returns import (
    calculate_daily_returns,
    calculate_log_returns,
    annualized_return,
    annualized_volatility
)


tickers = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "LINK-USD"
]


# Download prices
prices = download_prices(
    tickers,
    "2025-01-01",
    "2026-01-01"
)


# Calculate returns
daily_returns = calculate_daily_returns(prices)
log_returns = calculate_log_returns(prices)


print("PRICE DATA")
print(prices.head())

print("\nDAILY RETURNS")
print(daily_returns.head())

print("\nLOG RETURNS")
print(log_returns.head())


print("\nANNUALIZED RETURN")
print(annualized_return(daily_returns))


print("\nANNUALIZED VOLATILITY")
print(annualized_volatility(daily_returns))