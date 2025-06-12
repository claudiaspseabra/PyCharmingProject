import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
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

df = data[["response_time", "crime_duration", "Hour", "Year", "Crime Name1", "Police District Name"]].dropna()

top_crimes = df["Crime Name1"].value_counts().nlargest(10).index
df = df[df["Crime Name1"].isin(top_crimes)]
df = df.sample(n=10000, random_state=42)

df["Crime_Label"] = LabelEncoder().fit_transform(df["Crime Name1"])
df["District_Code"] = LabelEncoder().fit_transform(df["Police District Name"])

X = df[["response_time", "crime_duration", "Hour", "Year", "District_Code"]]
y = df["Crime_Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# MLP SINGLE LAYER
mlp_single = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
start = time.time()
mlp_single.fit(X_train, y_train)
end = time.time()
print(f"\nMLP Single Layer (sem PCA) - Tempo de treino: {end - start:.2f} segundos")

y_pred_single = mlp_single.predict(X_test)
print("MLP Single Layer - Classification Report:")
print(classification_report(y_test, y_pred_single))

print("MLP Single Layer - Confusion Matrix:")
cm_single = confusion_matrix(y_test, y_pred_single)
print(cm_single)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_single, annot=True, fmt='d', cmap='Blues')
plt.title("MLP Single Layer - Confusion Matrix (Crime Name1)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# MLP MULTI LAYER
mlp_multi = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
start = time.time()
mlp_multi.fit(X_train, y_train)
end = time.time()
print(f"\nMLP Multi Layer (sem PCA) - Tempo de treino: {end - start:.2f} segundos")

y_pred_multi = mlp_multi.predict(X_test)
print("MLP Multi Layer - Classification Report:")
print(classification_report(y_test, y_pred_multi))

print("MLP Multi Layer - Confusion Matrix:")
cm_multi = confusion_matrix(y_test, y_pred_multi)
print(cm_multi)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Greens')
plt.title("MLP Multi Layer - Confusion Matrix (Crime Name1)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# MLP SINGLE LAYER PCA
pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA: Reduzido de {X_scaled.shape[1]} para {X_pca.shape[1]} componentes")

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)

mlp_single_pca = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
start = time.time()
mlp_single_pca.fit(X_train_pca, y_train_pca)
end = time.time()
print(f"\nMLP Single Layer (com PCA) - Tempo de treino: {end - start:.2f} segundos")

y_pred_single_pca = mlp_single_pca.predict(X_test_pca)
print("MLP Single Layer with PCA - Classification Report:")
print(classification_report(y_test_pca, y_pred_single_pca))

print("MLP Single Layer with PCA - Confusion Matrix:")
cm_single_pca = confusion_matrix(y_test_pca, y_pred_single_pca)
print(cm_single_pca)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_single_pca, annot=True, fmt='d', cmap='Purples')
plt.title("MLP Single Layer with PCA - Confusion Matrix (Crime Name1)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# sem PCA

# MLP MULTI LAYER PCA
mlp_multi_pca = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
start = time.time()
mlp_multi_pca.fit(X_train_pca, y_train_pca)
end = time.time()
print(f"\nMLP Multi Layer (com PCA) - Tempo de treino: {end - start:.2f} segundos")

y_pred_multi_pca = mlp_multi_pca.predict(X_test_pca)
print("MLP Multi Layer with PCA - Classification Report:")
print(classification_report(y_test_pca, y_pred_multi_pca))

print("MLP Multi Layer with PCA - Confusion Matrix:")
cm_multi_pca = confusion_matrix(y_test_pca, y_pred_multi_pca)
print(cm_multi_pca)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_multi_pca, annot=True, fmt='d', cmap='Oranges')
plt.title("MLP Multi Layer with PCA - Confusion Matrix (Crime Name1)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
