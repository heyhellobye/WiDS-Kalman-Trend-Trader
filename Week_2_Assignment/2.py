import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pykalman import KalmanFilter

# =====================================================
# Step 1: Fetch MSFT historical data
# =====================================================
data = yf.download(
    "MSFT",
    start="2015-01-01",
    end="2025-01-01",
    auto_adjust=True,
    progress=False
)

df = data[["Close", "Volume"]].copy()

# =====================================================
# Step 2: Feature Engineering
# =====================================================
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df["ret_1"] = df["log_return"].shift(1)
df["ret_2"] = df["log_return"].shift(2)

df["ma_5"] = df["Close"].rolling(5).mean()
df["ma_20"] = df["Close"].rolling(20).mean()
df["ma_60"] = df["Close"].rolling(60).mean()

df["roc_10"] = df["Close"].pct_change(10)
df["vol_20"] = df["log_return"].rolling(20).std()

df = df.dropna()

# =====================================================
# Step 3: Kalman Filter (Time-varying regression)
# =====================================================
features = [
    "ret_1", "ret_2", "roc_10", "vol_20",
    "ma_5", "ma_20", "ma_60"
]

X = df[features].values
y = df["log_return"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_features = X_scaled.shape[1]

kf = KalmanFilter(
    transition_matrices=np.eye(n_features),
    observation_matrices=X_scaled.reshape(-1, 1, n_features),
    initial_state_mean=np.zeros(n_features),
    transition_covariance=0.01 * np.eye(n_features),
    observation_covariance=1.0
)

state_means, state_covs = kf.filter(y)

# =====================================================
# Plot Kalman-filtered parameters
# =====================================================
plt.figure(figsize=(10, 6))
for i in range(n_features):
    plt.plot(state_means[:, i], label=features[i])
plt.legend()
plt.title("Kalman-filtered Time-varying Parameters")
plt.show()

# =====================================================
# Step 4: Machine Learning Model
# =====================================================
df_kf = pd.DataFrame(
    state_means,
    index=df.index,
    columns=[f"beta_{f}" for f in features]
)

df_kf["target"] = df["Close"].shift(-1) / df["Close"]

df_kf = df_kf.dropna()

X_ml = df_kf.drop(columns=["target"]).values
y_ml = df_kf["target"].values

split = int(0.7 * len(df_kf))
X_train, X_test = X_ml[:split], X_ml[split:]
y_train, y_test = y_ml[:split], y_ml[split:]

ml_model = LinearRegression()
ml_model.fit(X_train, y_train)

df_kf["pred_ratio"] = ml_model.predict(X_ml)

# =====================================================
# Step 5: Trading Signal Generation
# =====================================================
threshold = 0.002

df_kf["signal"] = 0
df_kf.loc[df_kf["pred_ratio"] > 1 + threshold, "signal"] = 1
df_kf.loc[df_kf["pred_ratio"] < 1 - threshold, "signal"] = -1

# =====================================================
# Step 6: Backtesting
# =====================================================
df_kf["position"] = df_kf["signal"].shift(1).fillna(0)
df_kf["market_return"] = df["log_return"].loc[df_kf.index]
df_kf["strategy_return"] = df_kf["position"] * df_kf["market_return"]

transaction_cost = 0.001
df_kf["turnover"] = df_kf["position"].diff().abs()
df_kf["strategy_return"] -= transaction_cost * df_kf["turnover"]

df_kf["equity"] = (1 + df_kf["strategy_return"]).cumprod()
df_kf["bh_equity"] = (1 + df_kf["market_return"]).cumprod()

# =====================================================
# Step 7: Performance Metrics
# =====================================================
def sharpe_ratio(r):
    return np.sqrt(252) * r.mean() / r.std()

cumulative_return = df_kf["equity"].iloc[-1] - 1
sharpe = sharpe_ratio(df_kf["strategy_return"])

drawdown = df_kf["equity"] / df_kf["equity"].cummax() - 1
max_dd = drawdown.min()

win_ratio = (df_kf["strategy_return"] > 0).mean()

print("\n===== STRATEGY PERFORMANCE =====")
print("Cumulative Return:", cumulative_return)
print("Sharpe Ratio:", sharpe)
print("Max Drawdown:", max_dd)
print("Win Ratio:", win_ratio)

# =====================================================
# Plots
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(df_kf["equity"], label="Strategy")
plt.plot(df_kf["bh_equity"], label="Buy & Hold")
plt.legend()
plt.title("Equity Curve Comparison")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["Close"], label="MSFT Price")
plt.scatter(
    df_kf[df_kf["signal"] == 1].index,
    df.loc[df_kf[df_kf["signal"] == 1].index, "Close"],
    marker="^",
    color="green",
    label="Buy"
)
plt.scatter(
    df_kf[df_kf["signal"] == -1].index,
    df.loc[df_kf[df_kf["signal"] == -1].index, "Close"],
    marker="v",
    color="red",
    label="Sell"
)
plt.legend()
plt.title("Trading Signals")
plt.show()
