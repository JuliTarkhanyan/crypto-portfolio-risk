import numpy as np

TRADING_DAYS = 365


def calculate_daily_returns(prices):
    """
    Calculate simple daily percentage returns.
    """
    returns = prices.pct_change()
    return returns.dropna(how="all").dropna(axis=1, how="all")


def calculate_log_returns(prices):
    """
    Calculate daily logarithmic returns.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how="all").dropna(axis=1, how="all")


def annualized_return(returns):
    """
    Calculate annualized return from daily returns.
    """
    if len(returns) == 0:
        return np.nan

    return (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1


def annualized_volatility(returns):
    """
    Calculate annualized volatility.
    """
    return returns.std() * np.sqrt(TRADING_DAYS)