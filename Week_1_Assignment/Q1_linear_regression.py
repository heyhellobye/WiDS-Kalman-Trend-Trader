import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("linear_regression_dataset.csv")

X = df.drop(columns=["y"]).values
y = df["y"].values

n = X.shape[0]

# Add intercept
X = np.column_stack([np.ones(n), X])

# -----------------------------
# Closed-form OLS
# -----------------------------
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
beta_hat = XtX_inv @ X.T @ y

print("\nOLS coefficients (NumPy):")
print(beta_hat)

# -----------------------------
# sklearn comparison
# -----------------------------
lr = LinearRegression()
lr.fit(X[:, 1:], y)

print("\nsklearn coefficients:")
print(np.concatenate(([lr.intercept_], lr.coef_)))

# -----------------------------
# Predictions and residuals
# -----------------------------
y_hat = X @ beta_hat
residuals = y - y_hat

# -----------------------------
# Residual vs fitted
# -----------------------------
plt.figure()
plt.scatter(y_hat, residuals)
plt.axhline(0)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

# -----------------------------
# Q-Q plot
# -----------------------------
plt.figure()
stats.probplot(residuals, plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()

# -----------------------------
# Hat matrix and leverage
# -----------------------------
H = X @ XtX_inv @ X.T
leverage = np.diag(H)

print("\nTop 10 leverage points:")
print(np.argsort(leverage)[-10:])

# -----------------------------
# Cook's distance
# -----------------------------
p = X.shape[1]
mse = np.mean(residuals**2)
cooks_d = (residuals**2 / (p * mse)) * (leverage / (1 - leverage)**2)

print("\nTop 10 Cook's distance points:")
print(np.argsort(cooks_d)[-10:])
