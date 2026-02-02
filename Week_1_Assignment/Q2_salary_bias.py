import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_ind

# =========================================================
# Load dataset
# =========================================================
df = pd.read_csv("salary_dataset.csv")

# =========================================================
# EDA
# =========================================================
print("\nBasic Statistics:")
print(df.describe())

plt.figure()
sns.histplot(df["salary"], kde=True)
plt.title("Salary Distribution")
plt.show()

plt.figure()
sns.boxplot(x="gender", y="salary", data=df)
plt.title("Salary by Gender")
plt.show()

# =========================================================
# Handle missing values
# =========================================================
df = df.dropna()

# =========================================================
# Store gender BEFORE encoding (IMPORTANT)
# =========================================================
gender_original = df["gender"].copy()

# =========================================================
# Encode categorical variables
# =========================================================
cat_cols = df.select_dtypes(include="object").columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# =========================================================
# Features and target
# =========================================================
y = df["salary"]
X = df.drop(columns=["salary"])

# =========================================================
# Train-test split (stratified by gender)
# =========================================================
X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
    X,
    y,
    gender_original,
    test_size=0.3,
    random_state=42,
    stratify=gender_original
)

# =========================================================
# Train OLS Linear Regression
# =========================================================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================================================
# Predictions
# =========================================================
y_pred = model.predict(X_test)

# =========================================================
# Evaluation metrics
# =========================================================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)

# =========================================================
# Create test dataframe (ONE place)
# =========================================================
df_test = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred,
    "gender": gender_test.values
})

# =========================================================
# Residuals (DO THIS FIRST)
# =========================================================
df_test["residual"] = df_test["y_true"] - df_test["y_pred"]

# =========================================================
# Split by gender (AFTER residuals)
# =========================================================
male = df_test[df_test["gender"] == "Male"]
female = df_test[df_test["gender"] == "Female"]
other = df_test[df_test["gender"] == "Other"]

# =========================================================
# Fairness Metrics
# =========================================================
print("\nFairness Metrics:")

mean_pred_diff_mf = male["y_pred"].mean() - female["y_pred"].mean()
mae_diff_mf = mean_absolute_error(male["y_true"], male["y_pred"]) - \
              mean_absolute_error(female["y_true"], female["y_pred"])

print("Male vs Female:")
print("Mean Prediction Difference:", mean_pred_diff_mf)
print("MAE Difference:", mae_diff_mf)

if len(other) > 0:
    mean_pred_diff_mo = male["y_pred"].mean() - other["y_pred"].mean()
    mae_diff_mo = mean_absolute_error(male["y_true"], male["y_pred"]) - \
                  mean_absolute_error(other["y_true"], other["y_pred"])

    print("\nMale vs Other:")
    print("Mean Prediction Difference:", mean_pred_diff_mo)
    print("MAE Difference:", mae_diff_mo)

# =========================================================
# Residual Distribution Plot
# =========================================================
plt.figure()
sns.histplot(male["residual"], kde=True, label="Male", color="blue")
sns.histplot(female["residual"], kde=True, label="Female", color="red")
plt.legend()
plt.title("Residual Distribution by Gender")
plt.show()

# =========================================================
# T-test on residuals (Male vs Female)
# =========================================================
t_stat, p_val = ttest_ind(
    male["residual"],
    female["residual"],
    equal_var=False
)

print("\nT-test on residuals (Male vs Female):")
print("t-statistic:", t_stat)
print("p-value    :", p_val)

if p_val < 0.05:
    print("Conclusion: Statistically significant difference in residual means.")
else:
    print("Conclusion: No statistically significant difference in residual means.")
