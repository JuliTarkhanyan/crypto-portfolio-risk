from src.coingecko import download_coin_details


coins = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "LINK-USD",
]


for ticker in coins:

    print("\n")
    print("=" * 70)
    print(ticker)
    print("=" * 70)

    details = download_coin_details(ticker)

    print("\nName:")
    print(details["name"])

    print("\nSymbol:")
    print(details["symbol"])

    print("\nGenesis date:")
    print(details["genesis_date"])

    print("\nCategories:")
    print(details["categories"])

    print("\nHashing algorithm:")
    print(details["hashing_algorithm"])

    print("\nBlock time:")
    print(details["block_time"])

    print("\nDescription:")
    print(details["description"][:500])

    print("\nHomepage:")
    print(details["homepage"])

    print("\nWhitepaper:")
    print(details["whitepaper"])

    print("\nExplorer:")
    print(details["blockchain_explorer"])