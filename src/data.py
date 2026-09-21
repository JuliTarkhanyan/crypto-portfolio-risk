import yfinance as yf
import pandas as pd


def download_prices(tickers, start_date, end_date):
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No market data was returned.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            prices = data["Close"]
        else:
            prices = data.xs("Close", axis=1, level=0)
    else:
        if "Close" in data.columns:
            prices = data[["Close"]]
        else:
            prices = data

    prices = prices.sort_index()
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill()

    return prices