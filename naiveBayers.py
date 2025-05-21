import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df_crime = pd.read_csv(r"D:\Загрузки!\Crime (1).csv", low_memory=False)

df_crime = df_crime.dropna(subset=['Crime Name1'])

y = df_crime['Crime Name1']
X = df_crime.drop(columns=['Crime Name1']).select_dtypes(include=[np.number])
X = X.dropna()
y = y.loc[X.index]

# Train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("Naive Bayes classification report:")
print(classification_report(y_test, y_pred, zero_division=0))


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_model.classes_)
disp.plot()
plt.title("Confusion matrix (Naive Bayes)")
plt.show()