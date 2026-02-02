import yfinance as yf
import pandas as pd

def load_data():
    assets = {
        "BTC": "BTC-USD",
        "NIFTY": "^NSEI",
        "GOLD": "GLD"
    }

    prices = []

    for name, ticker in assets.items():
        df = yf.download(
            ticker,
            start="2015-01-01",
            end="2025-01-01",
            auto_adjust=True,
            progress=False
        )

        # Force Series (handles yfinance multi-column issue)
        series = df["Close"].iloc[:, 0]
        series.name = name
        prices.append(series)

    data = pd.concat(prices, axis=1).dropna()
    return data
