import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
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

svm_clf = SVC(kernel='linear', C=10, gamma='scale', class_weight='balanced')
cv_scores = cross_val_score(svm_clf, X_scaled, y, cv=5, scoring='accuracy')
print("Mean Accuracy (Cross-Validation):", cv_scores.mean())
print("Standard Deviation of Accuracy:", cv_scores.std())

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

start = time.time()
svm_clf.fit(X_train, y_train)
end = time.time()
print("Training time:", end - start)

y_pred = svm_clf.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM Classification (Crime Name1)")
plt.show()

# =============== PCA ===============

pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA: Reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} components")

svm_clf_pca = SVC(kernel='linear', C=10, gamma='scale', class_weight='balanced')
cv_scores_pca = cross_val_score(svm_clf_pca, X_pca, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy with PCA:", cv_scores_pca.mean())
print("Standard Deviation with PCA:", cv_scores_pca.std())

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)

start_pca = time.time()
svm_clf_pca.fit(X_train_pca, y_train_pca)
end_pca = time.time()
print("Training time with PCA:", end_pca - start_pca)

y_pred_pca = svm_clf_pca.predict(X_test_pca)

print("\nConfusion Matrix with PCA:")
print(confusion_matrix(y_test_pca, y_pred_pca))

print("\nClassification Report with PCA:")
print(classification_report(y_test_pca, y_pred_pca))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_pca, y_pred_pca), annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM Classification with PCA (Crime Name1)")
plt.show()
