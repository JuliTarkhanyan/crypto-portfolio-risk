import numpy as np

from src.data import download_prices

from src.returns import (
    calculate_daily_returns,
    annualized_return
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
    beta
)


# -----------------------------------
# 1. Download data
# -----------------------------------

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


# -----------------------------------
# 2. Calculate returns
# -----------------------------------

returns = calculate_daily_returns(prices)


# -----------------------------------
# 3. Portfolio weights
# -----------------------------------

weights = np.array([
    0.25,
    0.25,
    0.25,
    0.25
])


# -----------------------------------
# 4. Portfolio returns
# -----------------------------------

portfolio_daily_returns = portfolio_returns(
    returns,
    weights
)


# -----------------------------------
# 5. Risk calculations
# -----------------------------------

print("\n==============================")
print("PORTFOLIO RISK ANALYSIS")
print("==============================")

print(
    "\nAnnualized Return:",
    annualized_return(
        portfolio_daily_returns
    )
)

print(
    "Annualized Volatility:",
    portfolio_volatility(
        returns,
        weights
    )
)

print(
    "Portfolio Variance:",
    portfolio_variance(
        returns,
        weights
    )
)

print(
    "Historical VaR (95%):",
    historical_var(
        portfolio_daily_returns,
        0.95
    )
)

print(
    "Historical CVaR (95%):",
    historical_cvar(
        portfolio_daily_returns,
        0.95
    )
)

print(
    "Parametric VaR (95%):",
    parametric_var(
        portfolio_daily_returns,
        0.95
    )
)

print(
    "Maximum Drawdown:",
    maximum_drawdown(
        portfolio_daily_returns
    )
)

print(
    "Sharpe Ratio:",
    sharpe_ratio(
        portfolio_daily_returns
    )
)

print(
    "Sortino Ratio:",
    sortino_ratio(
        portfolio_daily_returns
    )
)

print(
    "Downside Deviation:",
    downside_deviation(
        portfolio_daily_returns
    )
)


# -----------------------------------
# 6. Covariance matrix
# -----------------------------------

print("\n==============================")
print("COVARIANCE MATRIX")
print("==============================")

print(
    covariance_matrix(returns)
)


# -----------------------------------
# 7. Correlation matrix
# -----------------------------------

print("\n==============================")
print("CORRELATION MATRIX")
print("==============================")

print(
    correlation_matrix(returns)
)


# -----------------------------------
# 8. BTC beta
# -----------------------------------

btc_returns = returns["BTC-USD"]

print("\n==============================")
print("BTC BETA")
print("==============================")

print(beta(portfolio_daily_returns,btc_returns))