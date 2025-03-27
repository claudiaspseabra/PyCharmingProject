import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
print("Variance Offence Code: ", variancia_offence)

# Zip Code
variancia_zip = crime["Zip Code"].var()
print("Variance Zip Code: ", variancia_zip)

# Victims
variancia_victims = crime["Victims"].var()
print("Variance Victims: ", variancia_victims)



#--------------------COVARIANCE------------------------

crime['Police_District_code'] = pd.factorize(crime['Police District Name'])[0]
crime['Crime_Name1_encoded'] = pd.factorize(crime['Crime Name1'])[0]
crime['Zip_Code_code'] = pd.factorize(crime['Zip Code'])[0]
crime['City_code'] = pd.factorize(crime['City'])[0]


# Covariances for Victims
covariance_victims_crime = crime['Victims'].cov(crime['Crime_Name1_encoded'])
covariance_victims_police_district = crime['Victims'].cov(crime['Police_District_code'])
covariance_victims_city = crime['Victims'].cov(crime['City_code'])
covariance_victims_zip = crime['Victims'].cov(crime['Zip_Code_code'])

# Covariances for Zip Code
covariance_zip_crime = crime['Zip_Code_code'].cov(crime['Crime_Name1_encoded'])
covariance_zip_police_district = crime['Zip_Code_code'].cov(crime['Police_District_code'])

# Covariances for Crime Name 1
covariance_crime_police_district = crime['Crime_Name1_encoded'].cov(crime['Police_District_code'])
covariance_crime_city = crime['Crime_Name1_encoded'].cov(crime['City_code'])
covariance_crime_zip = crime['Crime_Name1_encoded'].cov(crime['Zip_Code_code'])


print("\nCovariance between Victims and Crime Name 1:", covariance_victims_crime)
print("Covariance between Victims and Police District Name:", covariance_victims_police_district)
print("Covariance between Victims and City:", covariance_victims_city)
print("Covariance between Victims and Zip Code:", covariance_victims_zip)

print("\nCovariance between Zip Code and Crime Name 1:", covariance_zip_crime)
print("Covariance between Zip Code and Police District Name:", covariance_zip_police_district)

print("\nCovariance between Crime Name 1 and Police District Name:", covariance_crime_police_district)
print("Covariance between Crime Name 1 and City:", covariance_crime_city)
print("Covariance between Crime Name 1 and Zip Code:", covariance_crime_zip)



#----------------------GRAPH ANALYSIS---------------------------------


#------------------------JULIA-----------------------

#Crime by ZIP code (NAO DA. FAZER OUTRO TIPO DE GRAFICO)
crime_by_zip = crime.groupby(["Zip Code"])["Incident ID"].count()

crime_by_zip.plot(kind='bar', color=plt.cm.Pastel1.colors)

plt.title('Crime by Zip Code', fontsize=16, fontweight='bold')
plt.xlabel('Zip Code', fontsize=30)
plt.ylabel('Number of Incidents', fontsize=12)

plt.show()



#Average number of victims per crime (MUDAR A COR)
avg_victims_per_crime = crime.groupby(["Crime Name1"])["Victims"].mean()

plt.figure(figsize=(12, 6))
avg_victims_per_crime.sort_values(ascending=False).plot(kind='line', marker='o', color='dodgerblue', linewidth=2)

plt.title('Average Number of Victims per Crime', fontsize=16, fontweight='bold')
plt.xlabel('Crime Type', fontsize=12)
plt.ylabel('Average Number of Victims', fontsize=12)

plt.tight_layout()
plt.show()


#What crimes have the most victims (MUDAR A COR)
crime.groupby("Crime Name1")["Victims"].sum().sort_values(ascending=False).head(10).plot(kind='bar', title='Top Crimes by Victim Count')
plt.show()


#----------------------------EU---------------------

# Gráfico 1: Distribuição do Tipo de Crime (horizontal)

crime_count = crime['Crime Name1'].value_counts()
plt.figure(figsize=(10, 6))
crime_count.plot(kind='barh', color=sns.color_palette("Set2", len(crime_count)))
plt.title('Crime Type Distribution', fontsize=16)
plt.xlabel('Number of Incidents', fontsize=12)
plt.ylabel('Crime Type', fontsize=12)
plt.tight_layout()
plt.show()

# Gráfico 2: Crime Type Distribution for Each Police District (barras empilhadas)
crime_by_district = crime.groupby(['Police District Name', 'Crime Name1'])['Incident ID'].count().unstack()
crime_by_district.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='Set3')
plt.title('Crime Type Distribution for Each Police District', fontsize=16)
plt.xlabel('Police District Name', fontsize=12)
plt.ylabel('Number of Incidents', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico 3: Most Committed Crime (barra vertical)
most_committed_crime = crime['Crime Name1'].value_counts().head(1)
plt.figure(figsize=(8, 6))
most_committed_crime.plot(kind='bar', color='tomato')
plt.title('Most Committed Crime', fontsize=16)
plt.xlabel('Crime Type', fontsize=12)
plt.ylabel('Number of Incidents', fontsize=12)
plt.tight_layout()
plt.show()


#----------------JENIFER-----------



# Task 2: Most Committed Crime by City (Bar Plot - Seaborn)
plt.figure(figsize=(12, 6))
sns.barplot(
    x=crime["City"].value_counts().index,
    y=crime["City"].value_counts().values,
    hue=crime["City"].value_counts().index,
    palette="coolwarm",
    legend=False
)
plt.xlabel("City")
plt.ylabel("Number of Crimes")
plt.title("Most Committed Crime by City - Seaborn")
plt.xticks(rotation=45, ha='right')
plt.show()

# Task 4: Crime by Patrol Area (Beat) - Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(
    x=crime["Beat"].value_counts().index,
    y=crime["Beat"].value_counts().values,
    hue=crime["Beat"].value_counts().index,
    palette="viridis",
    legend=False
)
plt.xlabel("Patrol Area (Beat)")
plt.ylabel("Number of Crimes")
plt.title("Crime by Patrol Area (Beat) - Seaborn")
plt.xticks(rotation=45, ha='right')
plt.show()
