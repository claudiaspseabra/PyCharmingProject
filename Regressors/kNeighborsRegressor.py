import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import time

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
df_sample = df.sample(frac=1, random_state=42)

df_sample["Crime_Code"] = LabelEncoder().fit_transform(df_sample["Crime Name1"])
df_sample["District_Code"] = LabelEncoder().fit_transform(df_sample["Police District Name"])

Y = df_sample["response_time"]
X = df_sample[["crime_duration", "Hour", "Year", "Victims", "Crime_Code", "District_Code"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

knn_reg = KNeighborsRegressor(n_neighbors=5)

start = time.time()
knn_reg.fit(X_train, y_train)
end = time.time()

print("Training time (KNN):", end - start)

y_pred = knn_reg.predict(X_test)
print("MSE (KNN):", mean_squared_error(y_test, y_pred))
print("R2 (KNN):", r2_score(y_test, y_pred))

cv_scores = cross_val_score(knn_reg, X_scaled, Y, cv=5, scoring='r2')
print("Mean R2 (cross-validation, KNN):", cv_scores.mean())
print("Standard deviation (KNN):", cv_scores.std())

plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual response time")
plt.ylabel("Predicted response time")
plt.title("KNN Regression - Response Time Prediction")
plt.show()

# =============== PCA ===============

pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA: Reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} components")

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, Y, test_size=0.3, random_state=42)

knn_reg_pca = KNeighborsRegressor(n_neighbors=5)

start_pca = time.time()
knn_reg_pca.fit(X_train_pca, y_train_pca)
end_pca = time.time()

y_pred_pca = knn_reg_pca.predict(X_test_pca)
print("Training time (KNN + PCA):", end_pca - start_pca)
print("MSE (KNN + PCA):", mean_squared_error(y_test_pca, y_pred_pca))
print("R2 (KNN + PCA):", r2_score(y_test_pca, y_pred_pca))

cv_scores_pca = cross_val_score(knn_reg_pca, X_pca, Y, cv=5, scoring='r2')
print("Mean R2 (cross-validation, KNN + PCA):", cv_scores_pca.mean())
print("Standard deviation (KNN + PCA):", cv_scores_pca.std())

plt.scatter(y_test_pca, y_pred_pca, alpha=0.6, color='green')
plt.xlabel("Actual response time")
plt.ylabel("Predicted response time (PCA)")
plt.title("KNN Regression with PCA - Response Time Prediction")
plt.show()
