import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

data = pd.read_csv("Crime.csv", low_memory=False)
data = data[data["State"] == "MD"]

data["Start_Date_Time"] = pd.to_datetime(data["Start_Date_Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
data["End_Date_Time"] = pd.to_datetime(data["End_Date_Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
data["Dispatch Date / Time"] = pd.to_datetime(data["Dispatch Date / Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

data["response_time"] = (data["Dispatch Date / Time"] - data["Start_Date_Time"]).dt.total_seconds()
data["crime_duration"] = (data["End_Date_Time"] - data["Start_Date_Time"]).dt.total_seconds()
data["Hour"] = data["Start_Date_Time"].dt.hour
data["Year"] = data["Start_Date_Time"].dt.year

df = data[["response_time", "crime_duration", "Hour", "Year", "Crime Name2", "Police District Name"]].dropna()

top_crimes = df["Crime Name2"].value_counts().nlargest(10).index
df = df[df["Crime Name2"].isin(top_crimes)]

df = df.sample(n=10000, random_state=42)

df["Crime_Label"] = LabelEncoder().fit_transform(df["Crime Name2"])
df["District_Code"] = LabelEncoder().fit_transform(df["Police District Name"])

X = df[["response_time", "crime_duration", "Hour", "Year", "District_Code"]]
y = df["Crime_Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Single Layer
mlp_single = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)

start = time.time()
mlp_single.fit(X_train, y_train)
end = time.time()
print(f"MLP Single Layer - Training time: {end - start:.2f} segundos")

y_pred_single = mlp_single.predict(X_test)

print("MLP Single Layer - Classification Report:")
print(classification_report(y_test, y_pred_single))

print("MLP Single Layer - Confusion Matrix:")
cm_single = confusion_matrix(y_test, y_pred_single)
print(cm_single)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_single, annot=True, fmt='d', cmap='Blues')
plt.title("MLP Single Layer - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Multi Layer
mlp_multi = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)

start = time.time()
mlp_multi.fit(X_train, y_train)
end = time.time()
print(f"MLP Multi Layer - Training time: {end - start:.2f} segundos")

y_pred_multi = mlp_multi.predict(X_test)

print("MLP Multi Layer - Classification Report:")
print(classification_report(y_test, y_pred_multi))

print("MLP Multi Layer - Confusion Matrix:")
cm_multi = confusion_matrix(y_test, y_pred_multi)
print(cm_multi)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Greens')
plt.title("MLP Multi Layer - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
