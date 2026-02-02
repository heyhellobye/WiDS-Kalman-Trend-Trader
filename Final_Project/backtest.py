import pandas as pd
import numpy as np

def backtest(prices, weights, cost=0.001):
    returns = prices.pct_change().shift(-1)
    weights = weights.loc[returns.index]

    turnover = weights.diff().abs().sum(axis=1)
    tc = turnover * cost

    port_ret = (weights.drop(columns=["CASH"]) * returns).sum(axis=1) - tc
    equity = (1 + port_ret).cumprod()

    return equity, port_ret, turnover
