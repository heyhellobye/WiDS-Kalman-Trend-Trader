import numpy as np

def kalman_filter(series, q=1e-5, r=1e-2):
    n = len(series)
    x_hat = np.zeros(n)
    P = np.zeros(n)

    x_hat[0] = series.iloc[0]
    P[0] = 1.0

    for t in range(1, n):
        x_pred = x_hat[t-1]
        P_pred = P[t-1] + q

        K = P_pred / (P_pred + r)
        x_hat[t] = x_pred + K * (series.iloc[t] - x_pred)
        P[t] = (1 - K) * P_pred

    return x_hat
