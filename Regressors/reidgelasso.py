# Import required libraries
import time

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Loading the dataset
data = pd.read_csv("../Crime.csv", low_memory=False)

# Filtering only for Maryland
data = data[data["State"] == "MD"]

# Parsing datetime columns correctly (with fixed format)
data["Start_Date_Time"] = pd.to_datetime(data["Start_Date_Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
data["End_Date_Time"] = pd.to_datetime(data["End_Date_Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
data["Dispatch Date / Time"] = pd.to_datetime(data["Dispatch Date / Time"], format="%m/%d/%Y %I:%M:%S %p",
                                              errors="coerce")

#  Feature Engineering
data["response_time"] = (data["Dispatch Date / Time"] - data["Start_Date_Time"]).dt.total_seconds()
data["crime_duration"] = (data["End_Date_Time"] - data["Start_Date_Time"]).dt.total_seconds()
data["Hour"] = data["Start_Date_Time"].dt.hour
data["Year"] = data["Start_Date_Time"].dt.year

# Removing missing values
df = data[[
    "response_time", "crime_duration", "Hour", "Year", "Victims", "Crime Name1", "Police District Name"
]].dropna()

# Encode categorical columns
df["Crime_Code"] = LabelEncoder().fit_transform(df["Crime Name1"])
df["District_Code"] = LabelEncoder().fit_transform(df["Police District Name"])

# Defining target (Y) and features (X)
Y = df["response_time"]
X = df[["crime_duration", "Hour", "Year", "Victims", "Crime_Code", "District_Code"]]

# Standardizing the feature set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splited into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# Linear Regression
lr = LinearRegression()
start = time.time()
lr.fit(X_train, Y_train)
end = time.time()
print("Training time:", end - start)

Y_pred_lr = lr.predict(X_test)

# Ridge Regression with Cross-Validation
alphas = np.arange(0.1, 10, 0.1)
ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train, Y_train)
Y_pred_ridge = ridge.predict(X_test)

# Lasso Regression with Cross-Validation
lasso = LassoCV(alphas=alphas, max_iter=10000, cv=5)
lasso.fit(X_train, Y_train)
Y_pred_lasso = lasso.predict(X_test)

# Results
print("\nLinear Regression:")
print("R2 Score: {:.4f}".format(r2_score(Y_test, Y_pred_lr)))
print("MAE: {:.2f}".format(mean_absolute_error(Y_test, Y_pred_lr)))

print("\nRidge Regression:")
print("Best Alpha: {:.2f}".format(ridge.alpha_))
print("R2 Score: {:.4f}".format(r2_score(Y_test, Y_pred_ridge)))
print("MAE: {:.2f}".format(mean_absolute_error(Y_test, Y_pred_ridge)))

print("\nLasso Regression:")
print("Best Alpha: {:.2f}".format(lasso.alpha_))
print("R2 Score: {:.4f}".format(r2_score(Y_test, Y_pred_lasso)))
print("MAE: {:.2f}".format(mean_absolute_error(Y_test, Y_pred_lasso)))
