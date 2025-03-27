import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats

df = pd.read_excel("Crime Excel.xlsx")
df = df[["Victims", "Offence Code", "Crime Name1", "Start_Date_Time", "End_Date_Time"]]
df["Start_Date_Time"] = pd.to_datetime(df["Start_Date_Time"])
df["End_Date_Time"] = pd.to_datetime(df["End_Date_Time"])
df["Crime Duration"] = (df["End_Date_Time"] - df["Start_Date_Time"])
df = df.dropna()
df["Offence Code"] = df["Offence Code"].astype("category").cat.codes
df["Crime Name1"] = df["Crime Name1"].astype("category").cat.codes
print(df["Victims"].corr(df["Offence Code"], method="pearson"))
print(df["Victims"].corr(df["Crime Name1"], method="pearson"))
print(df["Crime Duration"].corr(df["Crime Name1"], method="pearson"))