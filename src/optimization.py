import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 365


# ==========================================
# Helper
# ==========================================

def normalize_weights(weights):
    """
    Make sure weights sum to 1.
    """

    weights = np.asarray(weights, dtype=float)

    total = weights.sum()

    if total <= 0:
        raise ValueError("Weights must have a positive sum.")

    return weights / total


# ==========================================
# 1. Equal Weight
# ==========================================

def equal_weight(returns):
    """
    Allocate the same percentage to every asset.
    """

    n = len(returns.columns)

    return np.ones(n) / n


# ==========================================
# 2. Inverse Volatility
# ==========================================

def inverse_volatility(returns):
    """
    Allocate more weight to assets
    with lower volatility.
    """

    volatility = (
        returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    inverse = 1 / volatility

    weights = inverse / inverse.sum()

    return weights.values


# ==========================================
# 3. Minimum Volatility
# ==========================================

def minimum_volatility(returns):
    """
    Find the portfolio with the lowest
    historical volatility.

    Constraints:
        sum(weights) = 1
        0 <= weight <= 1
    """

    covariance = (
        returns.cov().values
        * TRADING_DAYS
    )

    n = len(returns.columns)

    initial = np.ones(n) / n

    def objective(weights):

        variance = (
            weights.T
            @ covariance
            @ weights
        )

        return np.sqrt(
            max(variance, 0)
        )

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }

    bounds = [
        (0, 1)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(result.message)

    return normalize_weights(result.x)


# ==========================================
# 4. Maximum Sharpe
# ==========================================

def maximum_sharpe(
    returns,
    risk_free_rate=0
):
    """
    Find the portfolio with the
    maximum historical Sharpe ratio.
    """

    mean_returns = (
        returns.mean().values
        * TRADING_DAYS
    )

    covariance = (
        returns.cov().values
        * TRADING_DAYS
    )

    n = len(returns.columns)

    initial = np.ones(n) / n

    def objective(weights):

        portfolio_return = (
            weights @ mean_returns
        )

        portfolio_variance = (
            weights.T
            @ covariance
            @ weights
        )

        volatility = np.sqrt(
            max(portfolio_variance, 0)
        )

        if volatility == 0:
            return 1e6

        sharpe = (
            portfolio_return
            - risk_free_rate
        ) / volatility

        return -sharpe

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }

    bounds = [
        (0, 1)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(result.message)

    return normalize_weights(result.x)


# ==========================================
# 5. Risk Parity
# ==========================================

def risk_parity(returns):
    """
    Find weights where assets contribute
    approximately equally to portfolio risk.
    """

    covariance = (
        returns.cov().values
        * TRADING_DAYS
    )

    n = len(returns.columns)

    initial = np.ones(n) / n

    def objective(weights):

        portfolio_variance = (
            weights.T
            @ covariance
            @ weights
        )

        volatility = np.sqrt(
            max(portfolio_variance, 1e-12)
        )

        marginal_risk = (
            covariance @ weights
        )

        contribution = (
            weights
            * marginal_risk
            / volatility
        )

        target = volatility / n

        return np.sum(
            (contribution - target) ** 2
        )

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }

    bounds = [
        (0.0001, 1)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(result.message)

    return normalize_weights(result.x)


# ==========================================
# 6. Maximum Diversification
# ==========================================

def maximum_diversification(returns):
    """
    Maximize the diversification ratio:

        weighted average asset volatility
        --------------------------------
        portfolio volatility
    """

    covariance = (
        returns.cov().values
        * TRADING_DAYS
    )

    asset_volatility = (
        returns.std().values
        * np.sqrt(TRADING_DAYS)
    )

    n = len(returns.columns)

    initial = np.ones(n) / n

    def objective(weights):

        portfolio_volatility = np.sqrt(
            weights.T
            @ covariance
            @ weights
        )

        weighted_volatility = (
            weights
            @ asset_volatility
        )

        if portfolio_volatility == 0:
            return 1e6

        diversification_ratio = (
            weighted_volatility
            / portfolio_volatility
        )

        return -diversification_ratio

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }

    bounds = [
        (0, 1)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(result.message)

    return normalize_weights(result.x)


# ==========================================
# 7. Target Volatility
# ==========================================

def target_volatility(
    returns,
    target=0.30
):
    """
    Find the portfolio whose volatility
    is closest to the requested target.
    """

    covariance = (
        returns.cov().values
        * TRADING_DAYS
    )

    n = len(returns.columns)

    initial = np.ones(n) / n

    def objective(weights):

        variance = (
            weights.T
            @ covariance
            @ weights
        )

        volatility = np.sqrt(
            max(variance, 0)
        )

        return (
            volatility - target
        ) ** 2

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }

    bounds = [
        (0, 1)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(result.message)

    return normalize_weights(result.x)


# ==========================================
# Main function
# ==========================================

def calculate_weights(
    method,
    returns,
    target_vol=0.30
):

    if method == "Equal Weight":
        return equal_weight(returns)

    if method == "Inverse Volatility":
        return inverse_volatility(returns)

    if method == "Minimum Volatility":
        return minimum_volatility(returns)

    if method == "Risk Parity":
        return risk_parity(returns)

    if method == "Maximum Sharpe":
        return maximum_sharpe(returns)

    if method == "Maximum Diversification":
        return maximum_diversification(returns)

    if method == "Target Volatility":
        return target_volatility(
            returns,
            target_vol
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


# ==========================================
# Efficient Frontier
# ==========================================

def efficient_frontier(
    returns,
    points=50
):
    """
    Calculate portfolios along the
    efficient frontier.
    """

    mean_returns = (
        returns.mean().values
        * TRADING_DAYS
    )

    covariance = (
        returns.cov().values
        * TRADING_DAYS
    )

    n = len(returns.columns)

    min_return = mean_returns.min()
    max_return = mean_returns.max()

    target_returns = np.linspace(
        min_return,
        max_return,
        points
    )

    results = []

    for target in target_returns:

        initial = np.ones(n) / n

        def objective(weights):

            variance = (
                weights.T
                @ covariance
                @ weights
            )

            return np.sqrt(
                max(variance, 0)
            )

        constraints = [

            {
                "type": "eq",
                "fun": lambda w:
                    np.sum(w) - 1
            },

            {
                "type": "eq",
                "fun": lambda w:
                    w @ mean_returns - target
            }
        ]

        bounds = [
            (0, 1)
            for _ in range(n)
        ]

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if result.success:

            results.append({
                "Return": target,
                "Volatility": result.fun
            })

    return pd.DataFrame(results)