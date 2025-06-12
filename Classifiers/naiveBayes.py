import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

le = LabelEncoder()
df["Crime_Label"] = le.fit_transform(df["Crime Name1"])
df["District_Code"] = LabelEncoder().fit_transform(df["Police District Name"])

X = df[["response_time", "crime_duration", "Hour", "Year", "District_Code"]]
y = df["Crime_Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

nb_model = GaussianNB()

cv_scores = cross_val_score(nb_model, X_scaled, y, cv=5, scoring='accuracy')
print("Mean Accuracy (Cross-Validation):", cv_scores.mean())
print("Standard Deviation of Accuracy:", cv_scores.std())

start = time.time()
nb_model.fit(X_train, y_train)
end = time.time()

y_pred = nb_model.predict(X_test)

print("Training time:", end - start)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.inverse_transform(np.unique(y)))
disp.plot()
plt.title("Confusion Matrix (Naive Bayes)")
plt.show()

# PCA

pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA: Reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} components")

cv_scores_pca = cross_val_score(nb_model, X_pca, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy with PCA:", cv_scores_pca.mean())
print("Standard Deviation with PCA:", cv_scores_pca.std())

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)

start_pca = time.time()
nb_model.fit(X_train_pca, y_train_pca)
end_pca = time.time()

y_pred_pca = nb_model.predict(X_test_pca)

print("Training time with PCA:", end_pca - start_pca)
print("Accuracy with PCA:", accuracy_score(y_test_pca, y_pred_pca))
print("Classification Report with PCA:\n", classification_report(y_test_pca, y_pred_pca, zero_division=0))

cm_pca = confusion_matrix(y_test_pca, y_pred_pca)
disp_pca = ConfusionMatrixDisplay(confusion_matrix=cm_pca, display_labels=le.inverse_transform(np.unique(y)))
disp_pca.plot()
plt.title("Confusion Matrix (Naive Bayes with PCA)")
plt.show()
