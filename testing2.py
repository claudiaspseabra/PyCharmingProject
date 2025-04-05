import pandas as pd
from sklearn.preprocessing import LabelEncoder

crime = pd.read_csv("Crime.csv", low_memory=False)
encoder = LabelEncoder()
crime = crime[crime["State"] == "MD"]

#Número de vítimas & Crime Name 1

crime['CrimeCode'] = encoder.fit_transform(crime['Crime Name1'])
correlation_vict_crimename = crime[['CrimeCode', 'Victims']].corr(method='pearson')
print(correlation_vict_crimename)


#Duração do crime & Crime Name 1

crime["Start_Date_Time"] = pd.to_datetime(crime["Start_Date_Time"], format="%m/%d/%Y %I:%M:%S %p")
crime["End_Date_Time"] = pd.to_datetime(crime["End_Date_Time"], format="%m/%d/%Y %I:%M:%S %p")
crime["crime_duration"] = (crime["End_Date_Time"] - crime["Start_Date_Time"]).dt.total_seconds()

correlation_duration_crimename = crime[['crime_duration', 'CrimeCode']].corr(method='pearson')
print(correlation_duration_crimename)


# Número de vítimas & Offence Code

crime['Offence Code label'] = encoder.fit_transform(crime['Offence Code'])
correlation_vict_offcode = crime[['Victims', 'Offence Code label']].corr(method='pearson')
print(correlation_vict_offcode)