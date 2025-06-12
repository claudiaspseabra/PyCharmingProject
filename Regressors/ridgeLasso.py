import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA

data = pd.read_csv("../Crime.csv", low_memory=False)
data = data[data["State"] == "MD"]

data["Start_Date_Time"] = pd.to_datetime(data["Start_Date_Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
data["End_Date_Time"] = pd.to_datetime(data["End_Date_Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
data["Dispatch Date / Time"] = pd.to_datetime(data["Dispatch Date / Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

data["response_time"] = (data["Dispatch Date / Time"] - data["Start_Date_Time"]).dt.total_seconds()
data["crime_duration"] = (data["End_Date_Time"] - data["Start_Date_Time"]).dt.total_seconds()
data["Hour"] = data["Start_Date_Time"].dt.hour
data["Year"] = data["Start_Date_Time"].dt.year

df = data[["response_time", "crime_duration", "Hour", "Year", "Victims", "Crime Name1", "Police District Name"]].dropna()
df["Crime_Code"] = LabelEncoder().fit_transform(df["Crime Name1"])
df["District_Code"] = LabelEncoder().fit_transform(df["Police District Name"])

Y = df["response_time"]
X = df[["crime_duration", "Hour", "Year", "Victims", "Crime_Code", "District_Code"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

alphas = np.arange(0.1, 10, 0.1)

# Linear
start = time.time()
lr = LinearRegression()
lr.fit(X_train, Y_train)
end = time.time()
Y_pred_lr = lr.predict(X_test)
print("\nLinear Regression:")
print("Training time:", end - start)
print("R2:", r2_score(Y_test, Y_pred_lr))
print("MAE:", mean_absolute_error(Y_test, Y_pred_lr))

# Ridge
ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train, Y_train)
Y_pred_ridge = ridge.predict(X_test)
print("\nRidge Regression:")
print("Best alpha:", ridge.alpha_)
print("R2:", r2_score(Y_test, Y_pred_ridge))
print("MAE:", mean_absolute_error(Y_test, Y_pred_ridge))

# Lasso
lasso = LassoCV(alphas=alphas, max_iter=10000, cv=5)
lasso.fit(X_train, Y_train)
Y_pred_lasso = lasso.predict(X_test)
print("\nLasso Regression:")
print("Best alpha:", lasso.alpha_)
print("R2:", r2_score(Y_test, Y_pred_lasso))
print("MAE:", mean_absolute_error(Y_test, Y_pred_lasso))


print("\nPCA")

# PCA
pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduziu de {X_scaled.shape[1]} para {X_pca.shape[1]} componentes")

X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_pca, Y, test_size=0.3, random_state=42)

# Linear PCA
start = time.time()
lr_pca = LinearRegression()
lr_pca.fit(X_train_pca, Y_train_pca)
end = time.time()
Y_pred_lr_pca = lr_pca.predict(X_test_pca)
print("\nLinear Regression + PCA:")
print("Training time:", end - start)
print("R2:", r2_score(Y_test_pca, Y_pred_lr_pca))
print("MAE:", mean_absolute_error(Y_test_pca, Y_pred_lr_pca))

# Ridge PCA
ridge_pca = RidgeCV(alphas=alphas, cv=5)
ridge_pca.fit(X_train_pca, Y_train_pca)
Y_pred_ridge_pca = ridge_pca.predict(X_test_pca)
print("\nRidge Regression + PCA:")
print("Best alpha:", ridge_pca.alpha_)
print("R2:", r2_score(Y_test_pca, Y_pred_ridge_pca))
print("MAE:", mean_absolute_error(Y_test_pca, Y_pred_ridge_pca))

# Lasso PCA
lasso_pca = LassoCV(alphas=alphas, max_iter=10000, cv=5)
lasso_pca.fit(X_train_pca, Y_train_pca)
Y_pred_lasso_pca = lasso_pca.predict(X_test_pca)
print("\nLasso Regression + PCA:")
print("Best alpha:", lasso_pca.alpha_)
print("R2:", r2_score(Y_test_pca, Y_pred_lasso_pca))
print("MAE:", mean_absolute_error(Y_test_pca, Y_pred_lasso_pca))
