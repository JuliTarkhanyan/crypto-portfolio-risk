import numpy as np
from scipy.optimize import minimize


# ============================================================
# HELPERS
# ============================================================

def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)

    total = weights.sum()

    if total <= 0:
        return np.ones(len(weights)) / len(weights)

    return weights / total


def prepare_returns(returns):
    """
    Clean returns data before optimization.
    Keeps the same assets throughout the optimization process.
    """

    if returns is None:
        raise ValueError(
            "Returns data is required."
        )

    if returns.empty:
        raise ValueError(
            "Returns data is empty."
        )

    cleaned = returns.copy()

    # Remove columns that contain no usable data
    cleaned = cleaned.dropna(
        axis=1,
        how="all",
    )

    # Forward fill missing values
    cleaned = cleaned.ffill()

    # Remove rows that still contain missing values
    cleaned = cleaned.dropna()

    if cleaned.empty:
        raise ValueError(
            "No usable return observations found."
        )

    if len(cleaned.columns) == 0:
        raise ValueError(
            "No usable assets found."
        )

    return cleaned


# ============================================================
# EQUAL WEIGHT
# ============================================================

def equal_weight(returns):

    n_assets = len(returns.columns)

    if n_assets == 0:
        return np.array([])

    return np.ones(n_assets) / n_assets


# ============================================================
# INVERSE VOLATILITY
# ============================================================

def inverse_volatility(returns):

    volatility = returns.std()

    volatility = volatility.replace(
        0,
        np.nan,
    )

    inverse_vol = 1 / volatility

    inverse_vol = (
        inverse_vol
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    if inverse_vol.sum() == 0:
        return equal_weight(returns)

    weights = (
        inverse_vol
        / inverse_vol.sum()
    )

    return weights.values


# ============================================================
# PORTFOLIO VOLATILITY
# ============================================================

def portfolio_volatility(
    weights,
    covariance_matrix,
):

    annual_covariance = (
        covariance_matrix * 365
    )

    variance = (
        weights.T
        @ annual_covariance.values
        @ weights
    )

    return np.sqrt(
        max(
            variance,
            0,
        )
    )


# ============================================================
# MINIMUM VOLATILITY
# ============================================================

def minimum_volatility(returns):

    covariance_matrix = returns.cov()

    n_assets = len(
        returns.columns
    )

    initial_weights = (
        np.ones(n_assets)
        / n_assets
    )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1,
        }
    ]

    bounds = [
        (0, 1)
        for _ in range(n_assets)
    ]

    result = minimize(
        lambda weights:
            portfolio_volatility(
                weights,
                covariance_matrix,
            ),
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        return normalize_weights(
            result.x
        )

    return initial_weights


# ============================================================
# MAXIMUM SHARPE
# ============================================================

def maximum_sharpe(
    returns,
    risk_free_rate=0.0,
):

    expected_returns = (
        returns.mean()
        * 365
    )

    covariance_matrix = returns.cov()

    n_assets = len(
        returns.columns
    )

    initial_weights = (
        np.ones(n_assets)
        / n_assets
    )

    def negative_sharpe(weights):

        annual_return = (
            weights
            @ expected_returns.values
        )

        volatility = portfolio_volatility(
            weights,
            covariance_matrix,
        )

        if volatility <= 0:
            return 0

        sharpe = (
            annual_return
            - risk_free_rate
        ) / volatility

        return -sharpe

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1,
        }
    ]

    bounds = [
        (0, 1)
        for _ in range(n_assets)
    ]

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        return normalize_weights(
            result.x
        )

    return initial_weights


# ============================================================
# RISK PARITY
# ============================================================

def risk_parity(returns):

    covariance_matrix = (
        returns.cov().values
        * 365
    )

    n_assets = len(
        returns.columns
    )

    initial_weights = (
        np.ones(n_assets)
        / n_assets
    )

    def objective(weights):

        portfolio_variance = (
            weights.T
            @ covariance_matrix
            @ weights
        )

        if portfolio_variance <= 0:
            return 0

        marginal_contribution = (
            covariance_matrix
            @ weights
        )

        risk_contribution = (
            weights
            * marginal_contribution
        )

        target = (
            portfolio_variance
            / n_assets
        )

        return np.sum(
            (
                risk_contribution
                - target
            ) ** 2
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1,
        }
    ]

    bounds = [
        (0, 1)
        for _ in range(n_assets)
    ]

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        return normalize_weights(
            result.x
        )

    return initial_weights


# ============================================================
# MAXIMUM DIVERSIFICATION
# ============================================================

def maximum_diversification(returns):

    covariance_matrix = returns.cov()

    asset_volatility = returns.std()

    n_assets = len(
        returns.columns
    )

    initial_weights = (
        np.ones(n_assets)
        / n_assets
    )

    def negative_diversification(
        weights
    ):

        portfolio_vol = (
            portfolio_volatility(
                weights,
                covariance_matrix,
            )
        )

        weighted_asset_volatility = (
            weights
            @ (
                asset_volatility.values
                * np.sqrt(365)
            )
        )

        if portfolio_vol <= 0:
            return 0

        diversification_ratio = (
            weighted_asset_volatility
            / portfolio_vol
        )

        return -diversification_ratio

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1,
        }
    ]

    bounds = [
        (0, 1)
        for _ in range(n_assets)
    ]

    result = minimize(
        negative_diversification,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        return normalize_weights(
            result.x
        )

    return initial_weights


# ============================================================
# TARGET VOLATILITY
# ============================================================

def target_volatility(
    returns,
    target=0.40,
):

    covariance_matrix = returns.cov()

    n_assets = len(
        returns.columns
    )

    initial_weights = (
        np.ones(n_assets)
        / n_assets
    )

    def objective(weights):

        volatility = (
            portfolio_volatility(
                weights,
                covariance_matrix,
            )
        )

        return (
            volatility - target
        ) ** 2

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1,
        }
    ]

    bounds = [
        (0, 1)
        for _ in range(n_assets)
    ]

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        return normalize_weights(
            result.x
        )

    return initial_weights


# ============================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================

def calculate_weights(
    returns,
    method="Equal Weight",
    target_volatility_value=0.40,
):
    """
    Calculate portfolio weights using
    different portfolio optimization methods.

    Returns:
        numpy.ndarray
        One weight per cryptocurrency.
    """

    returns = prepare_returns(
        returns
    )

    n_assets = len(
        returns.columns
    )

    # --------------------------------------------------------
    # ONE ASSET
    # --------------------------------------------------------

    if n_assets == 1:
        return np.array([1.0])

    # --------------------------------------------------------
    # EQUAL WEIGHT
    # --------------------------------------------------------

    if method == "Equal Weight":

        weights = equal_weight(
            returns
        )

    # --------------------------------------------------------
    # INVERSE VOLATILITY
    # --------------------------------------------------------

    elif method == "Inverse Volatility":

        weights = inverse_volatility(
            returns
        )

    # --------------------------------------------------------
    # MINIMUM VOLATILITY
    # --------------------------------------------------------

    elif method == "Minimum Volatility":

        weights = minimum_volatility(
            returns
        )

    # --------------------------------------------------------
    # MAXIMUM SHARPE
    # --------------------------------------------------------

    elif method == "Maximum Sharpe":

        weights = maximum_sharpe(
            returns
        )

    # --------------------------------------------------------
    # RISK PARITY
    # --------------------------------------------------------

    elif method == "Risk Parity":

        weights = risk_parity(
            returns
        )

    # --------------------------------------------------------
    # MAXIMUM DIVERSIFICATION
    # --------------------------------------------------------

    elif method == "Maximum Diversification":

        weights = maximum_diversification(
            returns
        )

    # --------------------------------------------------------
    # TARGET VOLATILITY
    # --------------------------------------------------------

    elif method == "Target Volatility":

        weights = target_volatility(
            returns,
            target=target_volatility_value,
        )

    else:

        raise ValueError(
            f"Unknown optimization method: {method}"
        )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    weights = np.asarray(
        weights,
        dtype=float,
    ).reshape(-1)

    if len(weights) != n_assets:

        raise ValueError(
            "Optimization returned "
            f"{len(weights)} weights for "
            f"{n_assets} assets."
        )

    if not np.all(
        np.isfinite(weights)
    ):

        raise ValueError(
            "Optimization returned "
            "invalid weights."
        )

    weights = normalize_weights(
        weights
    )

    return weights