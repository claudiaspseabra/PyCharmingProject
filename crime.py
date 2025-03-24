import pandas as pd


#-----------------BASICS ANALYSIS-------------------

crime = pd.read_csv("Crime.csv", low_memory=False)
crime = crime[crime["State"] == "MD"]
print(crime.dtypes)

range_victims = crime['Victims'].max() - crime['Victims'].min()
print("Range of Victims:", range_victims)

# 3. Median 
median_cr_number = crime['CR Number'].median()
median_victims = crime['Victims'].median()

print("\nMedian of CR Number:", median_cr_number)
print("Median of Victims:", median_victims)

# 4. Mean 
mean_victims = crime['Victims'].mean()
print("\nMean of Victims:", mean_victims)

# Max
max_cr_number = crime['CR Number'].max()
max_victims = crime['Victims'].max()

print("Max of CR Number:", max_cr_number)
print("Max of Victims:", max_victims)

# Min
min_cr_number = crime['CR Number'].min()
min_victims = crime['Victims'].min()

print("Min of CR Number:", min_cr_number)
print("Min of Victims:", min_victims)

# 7. Quartiles
quartiles_victims = crime['Victims'].quantile([0.25, 0.5, 0.75])

print("\nQuartiles of Victims:")
print(quartiles_victims)


#-------------------------COUNTS--------------------------

# Crime Name 1
print("---------------------")
print("\n")
count_crime1 = crime["Crime Name1"].value_counts()
print(count_crime1)


# Crime Name 3
print("---------------------")
print("\n")
count_crime3 = crime["Crime Name3"].value_counts()
print(count_crime3)

# Police District Name
print("---------------------")
print("\n")
count_police_district = crime["Police District Name"].value_counts()
print(count_police_district)

# City
print("---------------------")
print("\n")
count_city = crime["City"].value_counts()
print(count_city)

# Zip Code
print("---------------------")
print("\n")
count_zip_code = crime["Zip Code"].value_counts()
print(count_zip_code)

# Agency
print("---------------------")
print("\n")
count_agency = crime["Agency"].value_counts()
print(count_agency)

# Place
print("---------------------")
print("\n")
count_place = crime["Place"].value_counts()
print(count_place)

# Street Type
print("---------------------")
print("\n")
count_strtype = crime["Street Type"].value_counts()
print(count_strtype)


#--------------------MODA---------------------

#Calcular todas as modas
print("All modes:")
print(crime.mode().iloc[0])



#-----------------VARIÂNCIA-------------------

# Offence Code (se tirar dá erro)
crime['Offence Code_encoded'] = pd.factorize(crime['Offence Code'])[0]

# Offence Code
variancia_offence = crime["Offence Code_encoded"].var()
print("Variância Offence Code: ", variancia_offence)

# Zip Code
variancia_zip = crime["Zip Code"].var()
print("Variância Zip Code: ", variancia_zip)

# Victims
variancia_victims = crime["Victims"].var()
print("Variância Victims: ", variancia_victims)



#--------------------COVARIÂNCIA------------------------

crime['Police_District_code'] = pd.factorize(crime['Police District Name'])[0]
crime['Crime_Name1_encoded'] = pd.factorize(crime['Crime Name1'])[0]
crime['Zip_Code_code'] = pd.factorize(crime['Zip Code'])[0]
crime['City_code'] = pd.factorize(crime['City'])[0]


# Covariâncias para Victims
covariance_victims_crime = crime['Victims'].cov(crime['Crime_Name1_encoded'])
covariance_victims_police_district = crime['Victims'].cov(crime['Police_District_code'])
covariance_victims_city = crime['Victims'].cov(crime['City_code'])
covariance_victims_zip = crime['Victims'].cov(crime['Zip_Code_code'])

# Covariâncias para Zip Code
covariance_zip_crime = crime['Zip_Code_code'].cov(crime['Crime_Name1_encoded'])
covariance_zip_police_district = crime['Zip_Code_code'].cov(crime['Police_District_code'])

# Covariâncias para Crime Name 1
covariance_crime_police_district = crime['Crime_Name1_encoded'].cov(crime['Police_District_code'])
covariance_crime_city = crime['Crime_Name1_encoded'].cov(crime['City_code'])
covariance_crime_zip = crime['Crime_Name1_encoded'].cov(crime['Zip_Code_code'])



print("\nCovariância entre Victims e Crime Name 1:", covariance_victims_crime)
print("Covariância entre Victims e Police District Name:", covariance_victims_police_district)
print("Covariância entre Victims e City:", covariance_victims_city)
print("Covariância entre Victims e Zip Code:", covariance_victims_zip)

print("\nCovariância entre Zip Code e Crime Name 1:", covariance_zip_crime)
print("Covariância entre Zip Code e Police District Name:", covariance_zip_police_district)

print("\nCovariância entre Crime Name 1 e Police District Name:", covariance_crime_police_district)
print("Covariância entre Crime Name 1 e City:", covariance_crime_city)
print("Covariância entre Crime Name 1 e Zip Code:", covariance_crime_zip)