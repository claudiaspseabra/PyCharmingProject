import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import time
from sklearn.decomposition import PCA

# Completed tasks:
# Analysis of the fit time per machine learning model (5%)
# Apply Principal Component Analysis and compare accuracy and execution time
# of the previous machine learning models. Use dimensionality reduction based on SVD. (5%)
# Perform Cross Validation (5%)

data = pd.read_csv("Crime.csv", low_memory=False)
data = data[data["State"] == "MD"]

data["Start_Date_Time"] = pd.to_datetime(data["Start_Date_Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
data["End_Date_Time"] = pd.to_datetime(data["End_Date_Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
data["Dispatch Date / Time"] = pd.to_datetime(data["Dispatch Date / Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

data["response_time"] = (data["Dispatch Date / Time"] - data["Start_Date_Time"]).dt.total_seconds()
data["crime_duration"] = (data["End_Date_Time"] - data["Start_Date_Time"]).dt.total_seconds()
data["Hour"] = data["Start_Date_Time"].dt.hour
data["Year"] = data["Start_Date_Time"].dt.year

df = data[["response_time", "crime_duration", "Hour", "Year", "Victims", "Crime Name1", "Police District Name"]].dropna()

df_sample = df.sample(frac = 1,random_state=42)

df_sample["Crime_Code"] = LabelEncoder().fit_transform(df_sample["Crime Name1"])
df_sample["District_Code"] = LabelEncoder().fit_transform(df_sample["Police District Name"])

'''
X = df_sample[["response_time", "crime_duration", "Hour", "Year", "Crime_Code", "District_Code"]]
y = df_sample["Victims"]
'''
Y = df_sample["response_time"]
X = df_sample[["crime_duration", "Hour", "Year", "Victims", "Crime_Code", "District_Code"]]

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

svr = SVR(kernel='linear', C=100, gamma=0.1, epsilon=0.2)

# training time
start = time.time()
svr.fit(X_train, y_train)
end = time.time()
print("Training time:", end - start)

y_pred = svr.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(svr, X_scaled, Y, cv=5, scoring='r2')
print("Mean R2 (cross-validation):", cv_scores.mean())
print("Standard deviation:", cv_scores.std())

plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual number of victims")
plt.ylabel("Predicted number of victims")
plt.title("SVM Regression - Victim Prediction")
plt.show()

# PCA + SVR

pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA: Reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} components")

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, Y, test_size=0.3, random_state=42)

svr_pca = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.2)

# training time with PCA
start_pca = time.time()
svr_pca.fit(X_train_pca, y_train_pca)
end_pca = time.time()
print("Training time (with PCA):", end_pca - start_pca)

y_pred_pca = svr_pca.predict(X_test_pca)

print("MSE (with PCA):", mean_squared_error(y_test_pca, y_pred_pca))
print("R2 (with PCA):", r2_score(y_test_pca, y_pred_pca))

# Cross-validation with PCA
cv_scores_pca = cross_val_score(svr_pca, X_pca, Y, cv=5, scoring='r2')
print("Mean R2 (cross-validation, PCA):", cv_scores_pca.mean())
print("Standard deviation (PCA):", cv_scores_pca.std())

plt.scatter(y_test_pca, y_pred_pca, alpha=0.6, color='green')
plt.xlabel("Actual number of victims")
plt.ylabel("Predicted number of victims (PCA)")
plt.title("SVM Regression with PCA - Victim Prediction")
plt.show()