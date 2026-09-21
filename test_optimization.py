from src.data import download_prices

from src.returns import (
    calculate_daily_returns
)

from src.optimization import (
    calculate_weights
)


# ==========================================
# Download data
# ==========================================

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


# ==========================================
# Returns
# ==========================================

returns = calculate_daily_returns(
    prices
)


# ==========================================
# Methods
# ==========================================

methods = [
    "Equal Weight",
    "Inverse Volatility",
    "Minimum Volatility",
    "Risk Parity",
    "Maximum Sharpe",
    "Maximum Diversification",
    "Target Volatility"
]


# ==========================================
# Calculate weights
# ==========================================

for method in methods:

    print("\n==============================")
    print(method)
    print("==============================")

    weights = calculate_weights(
        method,
        returns
    )

    for ticker, weight in zip(
        returns.columns,
        weights
    ):

        print(
            f"{ticker}: "
            f"{weight:.2%}"
        )

    print(
        "Total:",
        f"{weights.sum():.2%}"
    )