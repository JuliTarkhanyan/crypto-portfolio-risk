import numpy as np
import pandas as pd


TRADING_DAYS = 365


# ============================================================
# PORTFOLIO RETURNS
# ============================================================

def portfolio_returns(returns, weights):
    """
    Calculate daily portfolio returns.

    Parameters
    ----------
    returns : pandas.DataFrame
        Daily asset returns.

    weights : array-like
        Portfolio weights.

    Returns
    -------
    pandas.Series
        Daily portfolio returns.
    """

    if returns is None or returns.empty:
        return pd.Series(dtype=float)

    weights = np.asarray(
        weights,
        dtype=float
    )

    if len(weights) != len(returns.columns):
        raise ValueError(
            f"Number of weights ({len(weights)}) "
            f"does not match number of assets "
            f"({len(returns.columns)})."
        )

    return returns.dot(weights)


# ============================================================
# COVARIANCE MATRIX
# ============================================================

def covariance_matrix(returns):
    """
    Calculate annualized covariance matrix.

    Daily covariance is multiplied by 365.
    """

    if returns is None or returns.empty:
        return pd.DataFrame()

    clean_returns = returns.copy()

    clean_returns = clean_returns.dropna(
        axis=1,
        how="all"
    )

    clean_returns = (
        clean_returns
        .ffill()
        .dropna()
    )

    if clean_returns.empty:
        return pd.DataFrame()

    return (
        clean_returns.cov()
        * TRADING_DAYS
    )


# ============================================================
# CORRELATION MATRIX
# ============================================================

def correlation_matrix(returns):
    """
    Calculate Pearson correlation matrix.
    """

    if returns is None or returns.empty:
        return pd.DataFrame()

    clean_returns = returns.copy()

    clean_returns = clean_returns.dropna(
        axis=1,
        how="all"
    )

    clean_returns = (
        clean_returns
        .ffill()
        .dropna()
    )

    if clean_returns.empty:
        return pd.DataFrame()

    return clean_returns.corr()


# ============================================================
# BOTH RISK MATRICES
# ============================================================

def calculate_risk_matrices(returns):
    """
    Calculate covariance and correlation matrices.

    Returns
    -------
    covariance : pandas.DataFrame
        Annualized covariance matrix.

    correlation : pandas.DataFrame
        Pearson correlation matrix.
    """

    covariance = covariance_matrix(
        returns
    )

    correlation = correlation_matrix(
        returns
    )

    return covariance, correlation


# ============================================================
# PORTFOLIO VARIANCE
# ============================================================

def portfolio_variance(returns, weights):
    """
    Calculate annualized portfolio variance.

    Formula:

        w.T × Σ × w
    """

    covariance = covariance_matrix(
        returns
    )

    if covariance.empty:
        return np.nan

    weights = np.asarray(
        weights,
        dtype=float
    )

    if len(weights) != len(covariance.columns):
        raise ValueError(
            f"Number of weights ({len(weights)}) "
            f"does not match covariance matrix "
            f"dimension ({len(covariance.columns)})."
        )

    return (
        weights.T
        @ covariance.values
        @ weights
    )


# ============================================================
# PORTFOLIO VOLATILITY
# ============================================================

def portfolio_volatility(returns, weights):
    """
    Calculate annualized portfolio volatility.
    """

    variance = portfolio_variance(
        returns,
        weights
    )

    if np.isnan(variance):
        return np.nan

    return np.sqrt(
        max(variance, 0)
    )


# ============================================================
# HISTORICAL VaR
# ============================================================

def historical_var(
    returns,
    confidence=0.95
):
    """
    Historical Value at Risk.

    Returns the estimated daily loss threshold.
    """

    if returns is None or len(returns) == 0:
        return np.nan

    threshold = np.percentile(
        returns,
        (1 - confidence) * 100
    )

    return -threshold


# ============================================================
# HISTORICAL CVaR
# ============================================================

def historical_cvar(
    returns,
    confidence=0.95
):
    """
    Historical Conditional Value at Risk.

    Calculates the average loss beyond VaR.
    """

    if returns is None or len(returns) == 0:
        return np.nan

    threshold = np.percentile(
        returns,
        (1 - confidence) * 100
    )

    tail = returns[
        returns <= threshold
    ]

    if len(tail) == 0:
        return np.nan

    return -tail.mean()


# ============================================================
# PARAMETRIC VaR
# ============================================================

def parametric_var(
    returns,
    confidence=0.95
):
    """
    Parametric VaR assuming normally distributed returns.
    """

    if returns is None or len(returns) == 0:
        return np.nan

    mean = returns.mean()
    std = returns.std()

    z_scores = {
        0.90: 1.28155,
        0.95: 1.64485,
        0.99: 2.32635
    }

    if confidence not in z_scores:
        raise ValueError(
            "Confidence must be "
            "0.90, 0.95, or 0.99."
        )

    z = z_scores[confidence]

    return -(mean - z * std)


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

def maximum_drawdown(returns):
    """
    Calculate maximum drawdown.
    """

    if returns is None or len(returns) == 0:
        return np.nan

    wealth = (
        1 + returns
    ).cumprod()

    peak = wealth.cummax()

    drawdown = (
        wealth / peak
        - 1
    )

    return drawdown.min()


# ============================================================
# SHARPE RATIO
# ============================================================

def sharpe_ratio(
    returns,
    risk_free_rate=0
):
    """
    Calculate annualized Sharpe ratio.
    """

    if returns is None or len(returns) == 0:
        return np.nan

    annual_return = (
        (1 + returns).prod()
        ** (
            TRADING_DAYS
            / len(returns)
        )
        - 1
    )

    annual_volatility = (
        returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    if annual_volatility == 0:
        return np.nan

    return (
        annual_return
        - risk_free_rate
    ) / annual_volatility


# ============================================================
# SORTINO RATIO
# ============================================================

def sortino_ratio(
    returns,
    risk_free_rate=0
):
    """
    Calculate annualized Sortino ratio.
    """

    if returns is None or len(returns) == 0:
        return np.nan

    annual_return = (
        (1 + returns).prod()
        ** (
            TRADING_DAYS
            / len(returns)
        )
        - 1
    )

    downside = returns[
        returns < 0
    ]

    if len(downside) == 0:
        return np.nan

    downside_deviation = (
        downside.std()
        * np.sqrt(TRADING_DAYS)
    )

    if downside_deviation == 0:
        return np.nan

    return (
        annual_return
        - risk_free_rate
    ) / downside_deviation


# ============================================================
# DOWNSIDE DEVIATION
# ============================================================

def downside_deviation(returns):
    """
    Calculate annualized downside deviation.
    """

    if returns is None or len(returns) == 0:
        return 0

    negative_returns = returns[
        returns < 0
    ]

    if len(negative_returns) == 0:
        return 0

    return (
        negative_returns.std()
        * np.sqrt(TRADING_DAYS)
    )


# ============================================================
# BETA
# ============================================================

def beta(
    portfolio_returns,
    benchmark_returns
):
    """
    Calculate portfolio beta relative to a benchmark.
    """

    if (
        portfolio_returns is None
        or benchmark_returns is None
    ):
        return np.nan

    aligned = pd.concat(
        [
            portfolio_returns,
            benchmark_returns
        ],
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

    return (
        covariance
        / benchmark_variance
    )