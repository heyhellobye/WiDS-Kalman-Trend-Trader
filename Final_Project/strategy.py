import numpy as np
import pandas as pd

def allocate(trends):
    signals = trends.pct_change()
    signals = signals.clip(-0.02, 0.02)

    positive = signals.clip(lower=0)
    weights = positive.div(positive.sum(axis=1), axis=0).fillna(0)

    cash_weight = 1 - weights.sum(axis=1)
    weights["CASH"] = cash_weight

    return weights
