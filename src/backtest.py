import numpy as np
import pandas as pd


def buy_and_hold(
    prices,
    weights,
    initial_capital=10000,
):
    """
    Buy-and-hold backtest.

    The initial portfolio allocation is kept
    unchanged throughout the backtest.
    """

    prices = prices.copy().sort_index()

    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = weights / weights.sum()

    initial_prices = prices.iloc[0]

    initial_positions = (
        initial_capital
        * weights
        / initial_prices.values
    )

    portfolio_value = (
        prices
        * initial_positions
    ).sum(axis=1)

    daily_returns = (
        portfolio_value
        .pct_change()
        .fillna(0)
    )

    return pd.DataFrame(
        {
            "Portfolio Value":
                portfolio_value,
            "Daily Return":
                daily_returns,
        }
    )


def periodic_rebalance(
    prices,
    weights,
    initial_capital=10000,
    frequency="monthly",
):
    """
    Periodically rebalance portfolio.

    frequency:
        daily
        weekly
        monthly
        quarterly
    """

    prices = prices.copy().sort_index()

    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = weights / weights.sum()

    portfolio_value = pd.Series(
        index=prices.index,
        dtype=float,
    )

    current_value = initial_capital

    last_rebalance = None

    for date in prices.index:

        if last_rebalance is None:

            should_rebalance = True

        elif frequency == "daily":

            should_rebalance = True

        elif frequency == "weekly":

            should_rebalance = (
                date.isocalendar().week
                != last_rebalance.isocalendar().week
            )

        elif frequency == "monthly":

            should_rebalance = (
                date.month
                != last_rebalance.month
                or date.year
                != last_rebalance.year
            )

        elif frequency == "quarterly":

            current_quarter = (
                (date.month - 1) // 3
            )

            previous_quarter = (
                (last_rebalance.month - 1)
                // 3
            )

            should_rebalance = (
                current_quarter
                != previous_quarter
                or date.year
                != last_rebalance.year
            )

        else:

            raise ValueError(
                "Unsupported rebalance frequency."
            )

        if should_rebalance:

            allocation = (
                current_value
                * weights
            )

            shares = (
                allocation
                / prices.loc[date].values
            )

            last_rebalance = date

        portfolio_value.loc[date] = (
            prices.loc[date].values
            * shares
        ).sum()

        current_value = (
            portfolio_value.loc[date]
        )

    daily_returns = (
        portfolio_value
        .pct_change()
        .fillna(0)
    )

    return pd.DataFrame(
        {
            "Portfolio Value":
                portfolio_value,
            "Daily Return":
                daily_returns,
        }
    )


def backtest_metrics(backtest):
    """
    Calculate backtest performance metrics.
    """

    values = backtest[
        "Portfolio Value"
    ]

    returns = backtest[
        "Daily Return"
    ]

    total_return = (
        values.iloc[-1]
        / values.iloc[0]
        - 1
    )

    days = max(
        len(values),
        1,
    )

    annualized_return = (
        (1 + total_return)
        ** (365 / days)
        - 1
    )

    volatility = (
        returns.std()
        * np.sqrt(365)
    )

    cumulative_max = (
        values.cummax()
    )

    drawdown = (
        values / cumulative_max
        - 1
    )

    max_drawdown = drawdown.min()

    sharpe = (
        annualized_return
        / volatility
        if volatility > 0
        else np.nan
    )

    return {
        "Total Return":
            total_return,
        "Annualized Return":
            annualized_return,
        "Annualized Volatility":
            volatility,
        "Maximum Drawdown":
            max_drawdown,
        "Sharpe Ratio":
            sharpe,
    }