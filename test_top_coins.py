from src.coingecko import (
    download_market_chart_by_id,
    download_coin_details_by_id,
)


coin_id = "bitcoin"

prices = download_market_chart_by_id(
    coin_id,
    days=365
)

print("Historical prices:")
print(prices.head())

print("\nNumber of observations:")
print(len(prices))


details = download_coin_details_by_id(coin_id)

print("\nCoin:")
print(details["name"])

print("Symbol:")
print(details["symbol"])

print("Description:")
print(details["description"]["en"][:500])