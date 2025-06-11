import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

df["Crime_Label"] = LabelEncoder().fit_transform(df["Crime Name1"])
df["District_Code"] = LabelEncoder().fit_transform(df["Police District Name"])

X = df[["response_time", "crime_duration", "Hour", "Year", "District_Code"]]
y = df["Crime_Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

dt_clf = DecisionTreeClassifier(random_state=42)

start = time.time()
dt_clf.fit(X_train, y_train)
end = time.time()
print("Training time:", end - start)

y_pred = dt_clf.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree Classification (Crime Name1)")
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(dt_clf, feature_names=X.columns, class_names=[str(cls) for cls in sorted(y.unique())], filled=True)
plt.title("Decision Tree - Crime Classification (Crime Name1)")
plt.show()

cv_scores = cross_val_score(dt_clf, X_scaled, y, cv=5, scoring='accuracy')
print("\nMean Accuracy (Cross-Validation):", cv_scores.mean())
print("Standard Deviation of Accuracy:", cv_scores.std())

# PCA
pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA: Reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} components")

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)
dt_clf_pca = DecisionTreeClassifier(random_state=42)

start_pca = time.time()
dt_clf_pca.fit(X_train_pca, y_train_pca)
end_pca = time.time()
print("Training time with PCA:", end_pca - start_pca)

y_pred_pca = dt_clf_pca.predict(X_test_pca)

print("\nConfusion Matrix with PCA:")
print(confusion_matrix(y_test_pca, y_pred_pca))

print("\nClassification Report with PCA:")
print(classification_report(y_test_pca, y_pred_pca))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_pca, y_pred_pca), annot=True, fmt='d', cmap='Purples')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree with PCA (Crime Name1)")
plt.show()
