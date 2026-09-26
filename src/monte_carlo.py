import numpy as np
import pandas as pd


def monte_carlo_simulation(
    historical_returns,
    initial_capital=10000,
    days=365,
    simulations=1000,
    random_seed=None,
):
    """
    Run a Monte Carlo simulation using historical bootstrap sampling.

    Each simulated daily return is randomly sampled from the portfolio's
    historical daily returns.

    Parameters
    ----------
    historical_returns : pandas.Series
        Historical daily portfolio returns.

    initial_capital : float
        Starting portfolio value.

    days : int
        Number of days to simulate.

    simulations : int
        Number of simulation paths.

    random_seed : int or None
        Optional seed for reproducible results.

    Returns
    -------
    numpy.ndarray
        Array with shape (simulations, days).
        Each row represents one simulated portfolio path.
    """

    historical_returns = pd.Series(
        historical_returns
    ).dropna()

    if historical_returns.empty:
        raise ValueError(
            "Historical returns are empty."
        )

    if initial_capital <= 0:
        raise ValueError(
            "Initial capital must be greater than zero."
        )

    if days <= 0:
        raise ValueError(
            "Simulation days must be greater than zero."
        )

    if simulations <= 0:
        raise ValueError(
            "Number of simulations must be greater than zero."
        )

    # Convert returns to NumPy array
    historical_returns = historical_returns.to_numpy(
        dtype=float
    )

    # Random number generator
    rng = np.random.default_rng(
        random_seed
    )

    # Randomly sample historical daily returns
    simulated_returns = rng.choice(
        historical_returns,
        size=(simulations, days),
        replace=True,
    )

    # Convert returns into portfolio values
    simulated_values = (
        initial_capital
        * np.cumprod(
            1 + simulated_returns,
            axis=1
        )
    )

    return simulated_values


def monte_carlo_summary(
    simulated_values,
):
    """
    Calculate summary statistics for Monte Carlo simulations.
    """

    simulated_values = np.asarray(
        simulated_values,
        dtype=float
    )

    if simulated_values.ndim != 2:
        raise ValueError(
            "Simulated values must be a 2-dimensional array."
        )

    if simulated_values.shape[0] == 0:
        raise ValueError(
            "No simulation results available."
        )

    final_values = simulated_values[:, -1]

    summary = {
        "5th Percentile": np.percentile(
            final_values,
            5
        ),

        "25th Percentile": np.percentile(
            final_values,
            25
        ),

        "Median": np.percentile(
            final_values,
            50
        ),

        "75th Percentile": np.percentile(
            final_values,
            75
        ),

        "95th Percentile": np.percentile(
            final_values,
            95
        ),
    }

    return summary


def probability_of_loss(
    simulated_values,
    initial_capital,
):
    """
    Calculate the percentage of simulations that
    finish below the initial capital.
    """

    simulated_values = np.asarray(
        simulated_values,
        dtype=float
    )

    final_values = simulated_values[:, -1]

    probability = np.mean(
        final_values < initial_capital
    )

    return probability


def probability_of_profit(
    simulated_values,
    initial_capital,
):
    """
    Calculate the percentage of simulations that
    finish above the initial capital.
    """

    simulated_values = np.asarray(
        simulated_values,
        dtype=float
    )

    final_values = simulated_values[:, -1]

    probability = np.mean(
        final_values > initial_capital
    )

    return probability


def probability_of_target(
    simulated_values,
    target_value,
):
    """
    Calculate the percentage of simulations that
    reach or exceed a target portfolio value
    at the end of the simulation.
    """

    simulated_values = np.asarray(
        simulated_values,
        dtype=float
    )

    final_values = simulated_values[:, -1]

    probability = np.mean(
        final_values >= target_value
    )

    return probability


def monte_carlo_statistics(
    simulated_values,
    initial_capital,
):
    """
    Return a complete set of Monte Carlo statistics.
    """

    simulated_values = np.asarray(
        simulated_values,
        dtype=float
    )

    final_values = simulated_values[:, -1]

    median_value = np.median(
        final_values
    )

    mean_value = np.mean(
        final_values
    )

    minimum_value = np.min(
        final_values
    )

    maximum_value = np.max(
        final_values
    )

    return {
        "Mean Final Value": mean_value,

        "Median Final Value": median_value,

        "Minimum Final Value": minimum_value,

        "Maximum Final Value": maximum_value,

        "Probability of Loss": (
            probability_of_loss(
                simulated_values,
                initial_capital,
            )
        ),

        "Probability of Profit": (
            probability_of_profit(
                simulated_values,
                initial_capital,
            )
        ),
    }