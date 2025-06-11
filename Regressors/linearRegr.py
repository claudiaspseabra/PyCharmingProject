import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("../Crime.csv", low_memory=False)

df_numeric = df.select_dtypes(include=[np.number])

df_numeric = df_numeric.dropna()

X = df_numeric.drop(columns=['Victims'])
y = df_numeric['Victims']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("The model performance using score:")
print("--------------------------------------")
print(model.score(X_test, y_test))

y_test_predict = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
r2 = r2_score(y_test, y_test_predict)

print("The model performance using R2")
print("--------------------------------------")
print(f'R2 score is {r2}')
print(f'RMSE is {rmse}')
