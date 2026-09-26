import numpy as np
import pandas as pd


def normalize_weights(weights):
    """
    Normalize portfolio weights so that they sum to 1.
    """

    weights = np.asarray(weights, dtype=float)

    total = weights.sum()

    if total <= 0:
        raise ValueError(
            "Portfolio weights must have a positive sum."
        )

    return weights / total


def equal_weights(n_assets):
    """
    Create equal portfolio weights.
    """

    if n_assets <= 0:
        return np.array([])

    return np.ones(n_assets) / n_assets


def calculate_portfolio_value(
    prices,
    weights,
    initial_capital=10000,
):
    """
    Calculate portfolio value over time.

    Uses buy-and-hold weights based on the
    initial prices.
    """

    prices = prices.copy()

    weights = normalize_weights(weights)

    if len(weights) != len(prices.columns):
        raise ValueError(
            "Number of weights must match "
            "number of assets."
        )

    initial_prices = prices.iloc[0]

    shares = (
        initial_capital * weights
        / initial_prices.values
    )

    portfolio_value = (
        prices * shares
    ).sum(axis=1)

    return portfolio_value


def calculate_portfolio_returns(
    returns,
    weights,
):
    """
    Calculate daily portfolio returns.
    """

    weights = normalize_weights(weights)

    if len(weights) != len(returns.columns):
        raise ValueError(
            "Number of weights must match "
            "number of assets."
        )

    return returns.dot(weights)


def portfolio_statistics(
    returns,
    weights,
):
    """
    Calculate basic portfolio statistics.
    """

    portfolio_ret = calculate_portfolio_returns(
        returns,
        weights,
    )

    annual_return = (
        (1 + portfolio_ret.mean()) ** 365
    ) - 1

    annual_volatility = (
        portfolio_ret.std() * np.sqrt(365)
    )

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
    }