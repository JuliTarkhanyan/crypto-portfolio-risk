from src.coingecko import (
    download_crypto_info,
    download_market_chart,
)

from src.crypto_analysis import (
    supply_analysis,
    performance_analysis,
    volatility_analysis,
    drawdown_analysis,
    ath_analysis,
    trend_analysis,
    classify_trend,
    prepare_price_chart,
    prepare_rsi_chart,
    prepare_drawdown_chart,
)


ticker = "BTC-USD"


# --------------------------------
# Current market data
# --------------------------------

info = download_crypto_info(
    [ticker]
)

coin = info.iloc[0]


# --------------------------------
# Historical prices
# --------------------------------

history = download_market_chart(
    ticker,
    days=365
)

prices = history["price"]


# --------------------------------
# Supply analysis
# --------------------------------

supply = supply_analysis(
    circulating_supply=coin[
        "Circulating Supply"
    ],

    total_supply=coin[
        "Total Supply"
    ],

    max_supply=coin[
        "Max Supply"
    ],

    price=coin[
        "Price"
    ],

    market_cap=coin[
        "Market Cap"
    ],
)


print("\nTOKENOMICS")
print("=" * 60)

for key, value in supply.items():
    print(f"{key}: {value}")


# --------------------------------
# Performance
# --------------------------------

performance = performance_analysis(
    prices
)

print("\nPERFORMANCE")
print("=" * 60)

for key, value in performance.items():
    print(f"{key}: {value}")


# --------------------------------
# Volatility
# --------------------------------

volatility = volatility_analysis(
    prices
)

print("\nVOLATILITY")
print("=" * 60)

for key, value in volatility.items():
    print(f"{key}: {value}")


# --------------------------------
# Drawdown
# --------------------------------

drawdown = drawdown_analysis(
    prices
)

print("\nDRAWDOWN")
print("=" * 60)

for key, value in drawdown.items():
    print(f"{key}: {value}")


# --------------------------------
# ATH
# --------------------------------

ath = ath_analysis(
    current_price=coin["Price"],
    ath=coin["ATH"],
)

print("\nATH")
print("=" * 60)

for key, value in ath.items():
    print(f"{key}: {value}")

trend = trend_analysis(
    prices
)

print("\nTREND & TECHNICAL ANALYSIS")
print("=" * 60)

for key, value in trend.items():
    print(f"{key}: {value}")


trend_description = classify_trend(
    trend
)

print("\nTREND")
print("=" * 60)

print(trend_description)

price_chart = prepare_price_chart(
    prices
)

print("\nPRICE CHART DATA")
print("=" * 60)

print(
    price_chart.tail()
)


rsi_chart = prepare_rsi_chart(
    prices
)

print("\nRSI CHART DATA")
print("=" * 60)

print(
    rsi_chart.tail()
)


drawdown_chart = prepare_drawdown_chart(
    prices
)

print("\nDRAWDOWN CHART DATA")
print("=" * 60)

print(
    drawdown_chart.tail()
)