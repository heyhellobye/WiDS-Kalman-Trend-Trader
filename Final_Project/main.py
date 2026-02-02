import pandas as pd
from data import load_data
from kalman import kalman_filter
from strategy import allocate
from backtest import backtest
from metrics import performance_stats

prices = load_data()

trends = pd.DataFrame({
    asset: kalman_filter(prices[asset])
    for asset in prices.columns
}, index=prices.index)

weights = allocate(trends)
equity, returns, turnover = backtest(prices, weights)

stats = performance_stats(returns)

print("Annual Return:", stats[0])
print("Volatility:", stats[1])
print("Sharpe:", stats[2])
print("Max Drawdown:", stats[3])
