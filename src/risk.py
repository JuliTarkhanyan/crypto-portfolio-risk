import numpy as np
import pandas as pd

TRADING_DAYS = 365


def portfolio_returns(returns, weights):
    """
    Calculate daily portfolio returns.
    """
    weights = np.asarray(weights, dtype=float)

    return returns.dot(weights)


def covariance_matrix(returns):
    """
    Calculate annualized covariance matrix.
    """
    return returns.cov() * TRADING_DAYS


def correlation_matrix(returns):
    """
    Calculate correlation matrix.
    """
    return returns.corr()


def portfolio_variance(returns, weights):
    """
    Calculate annualized portfolio variance.

    Formula:
        w.T × Σ × w
    """
    covariance = covariance_matrix(returns)

    weights = np.asarray(weights, dtype=float)

    return weights.T @ covariance.values @ weights


def portfolio_volatility(returns, weights):
    """
    Calculate annualized portfolio volatility.
    """
    variance = portfolio_variance(returns, weights)

    return np.sqrt(max(variance, 0))


def historical_var(returns, confidence=0.95):
    """
    Historical Value at Risk.

    Returns the estimated daily loss threshold.
    """
    threshold = np.percentile(
        returns,
        (1 - confidence) * 100
    )

    return -threshold


def historical_cvar(returns, confidence=0.95):
    """
    Historical Conditional Value at Risk.

    Calculates the average loss beyond VaR.
    """
    threshold = np.percentile(
        returns,
        (1 - confidence) * 100
    )

    tail = returns[returns <= threshold]

    if len(tail) == 0:
        return np.nan

    return -tail.mean()


def parametric_var(returns, confidence=0.95):
    """
    Parametric VaR assuming normally distributed returns.
    """

    mean = returns.mean()
    std = returns.std()

    z_scores = {
        0.90: 1.28155,
        0.95: 1.64485,
        0.99: 2.32635
    }

    if confidence not in z_scores:
        raise ValueError(
            "Confidence must be 0.90, 0.95, or 0.99."
        )

    z = z_scores[confidence]

    return -(mean - z * std)


def maximum_drawdown(returns):
    """
    Calculate maximum drawdown.
    """

    wealth = (1 + returns).cumprod()

    peak = wealth.cummax()

    drawdown = wealth / peak - 1

    return drawdown.min()


def sharpe_ratio(
    returns,
    risk_free_rate=0
):
    """
    Calculate annualized Sharpe ratio.
    """

    if len(returns) == 0:
        return np.nan

    annual_return = (
        (1 + returns).prod()
        ** (TRADING_DAYS / len(returns))
        - 1
    )

    annual_volatility = (
        returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    if annual_volatility == 0:
        return np.nan

    return (
        annual_return - risk_free_rate
    ) / annual_volatility


def sortino_ratio(
    returns,
    risk_free_rate=0
):
    """
    Calculate annualized Sortino ratio.
    """

    if len(returns) == 0:
        return np.nan

    annual_return = (
        (1 + returns).prod()
        ** (TRADING_DAYS / len(returns))
        - 1
    )

    downside = returns[returns < 0]

    if len(downside) == 0:
        return np.nan

    downside_deviation = (
        downside.std()
        * np.sqrt(TRADING_DAYS)
    )

    if downside_deviation == 0:
        return np.nan

    return (
        annual_return - risk_free_rate
    ) / downside_deviation


def downside_deviation(returns):
    """
    Calculate annualized downside deviation.
    """

    negative_returns = returns[returns < 0]

    if len(negative_returns) == 0:
        return 0

    return (
        negative_returns.std()
        * np.sqrt(TRADING_DAYS)
    )


def beta(
    portfolio_returns,
    benchmark_returns
):
    """
    Calculate portfolio beta relative to a benchmark.
    """

    aligned = pd.concat(
        [portfolio_returns, benchmark_returns],
        axis=1
    ).dropna()

    if len(aligned) < 2:
        return np.nan

    covariance = np.cov(
        aligned.iloc[:, 0],
        aligned.iloc[:, 1]
    )[0, 1]

    benchmark_variance = np.var(
        aligned.iloc[:, 1]
    )

    if benchmark_variance == 0:
        return np.nan

    return covariance / benchmark_variance