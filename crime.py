import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
import numpy as np


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

# CRIMES BY ZIP CODE
crime_by_zip = crime.groupby("Zip Code")["Incident ID"].count()

plt.figure(figsize=(12, 8))
plt.scatter(crime_by_zip.index, crime_by_zip.values, c=crime_by_zip.values, cmap='coolwarm', s=100)

plt.title('Crime by Zip Code', fontsize=16, fontweight='bold')
plt.xlabel('Zip Code', fontsize=14)
plt.ylabel('Number of Crimes', fontsize=14)

plt.colorbar(label='Number of Crimes')
plt.show()


#AVERAGE NUMBER OF VICTIMS PER CRIME
avg_victims_per_crime = crime.groupby("Crime Name1")["Victims"].mean().sort_values(ascending=False)
categories = avg_victims_per_crime.index
values = avg_victims_per_crime.values


N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

# Create the radar graph
fig, ax = plt.subplots(figsize=(8, 8), dpi=120, subplot_kw=dict(polar=True))
ax.fill(angles, values, color='dodgerblue', alpha=0.25)
ax.plot(angles, values, color='dodgerblue', linewidth=2)

ax.set_yticklabels([])
ax.set_xticks(angles)
ax.set_xticklabels(categories, fontsize=10, ha='right')

#adding the average number of victims
for i, value in enumerate(values):
    ax.text(angles[i], value + 0.1, f'{value:.2f}', horizontalalignment='center', size=10, color='dodgerblue', weight='semibold')

plt.title('Average Number of Victims per Crime', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()



#WHAT CRIMES HAVE THE MOST VICTIMS
top_crimes = crime.groupby("Crime Name1")["Victims"].sum().sort_values(ascending=False).head(10)
top_crimes.plot(kind='area', stacked=True, color='skyblue', alpha=0.7, title='Distribuição das Vítimas por Crime')

plt.ylabel('Número de Vítimas')
plt.show()


#----------------------------Joana---------------------

# Gráfico 1: Distribuição do Tipo de Crime
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


#Distribution of Crime Type (each value in Crime Name1)
crime_counts = crime["Crime Name1"].value_counts()

plt.figure(figsize=(10, 6))
colors = plt.cm.tab10.colors
plt.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Distribution of Crime Type")
plt.axis('equal')
plt.show()


#Crime Type Distribution for Each Police District
crime_pivot = crime.pivot_table(index="Police District Name", columns="Crime Name1", aggfunc="size", fill_value=0)

plt.figure(figsize=(12, 6))
sns.heatmap(crime_pivot, cmap="Blues", annot=False, linewidths=0.5)
plt.title("Crime Type Distribution for Each Police District")
plt.xlabel("Crime Type")
plt.ylabel("Police District")
plt.xticks(rotation=45)
plt.show()


#----------------JENIFER-----------

# Most Committed Crime by City
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
plt.title("Most Committed Crime by City")
plt.xticks(rotation=45, ha='right')
plt.show()



# Crime by Patrol Area (using frequency Polygon)
crime_counts = crime["Beat"].value_counts()

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(
    x=crime_counts.index,
    y=crime_counts.values,
    marker="o",
    color='purple',
    linewidth=2,
    markersize=8
)
plt.fill_between(crime_counts.index, crime_counts.values, color="purple", alpha=0.2)

plt.title("Crime by Patrol Area")
plt.xlabel("Patrol Area (Beat)")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45, ha='right')
plt.show()


#---------------------------FRANCISCO-------------------

# NOVAS FEATURES:Duração do crime
crime["Start_Date_Time"] = pd.to_datetime(crime["Start_Date_Time"], format="%m/%d/%Y %I:%M:%S %p")
crime["End_Date_Time"] = pd.to_datetime(crime["End_Date_Time"], format="%m/%d/%Y %I:%M:%S %p")
crime["crime_duration"] = (crime["End_Date_Time"] - crime["Start_Date_Time"]).dt.total_seconds()

# Média da duração do crime
print("Duration mean")
mean_crime_duration = crime["crime_duration"].mean()
print("Média da duração do crime: ",mean_crime_duration)

# Mediana da duração do crime
print("Duration median")
median_crime_duration = crime["crime_duration"].median()
print("Mediana da duração do crime: ",median_crime_duration)

# Variância da duração do crime
print("Duration variance")
var_crime_duration = crime["crime_duration"].var()
print("Variância da duração do crime: ",var_crime_duration)

# Minimo da duração do crime
print("Min duration")
min_crime_duration = crime["crime_duration"].min()
print("Minimo da duração do crime: ",min_crime_duration)

print("Max duration")
max_crime_duration = crime["crime_duration"].max()
print("Máximo da duração do crime: ",max_crime_duration)

print("Duration quartiles")
quartile25 = crime["crime_duration"].quantile(0.25)
quartile50 = crime["crime_duration"].quantile(0.50)
quartile75 = crime["crime_duration"].quantile(0.75)
print("Quartile 25:",quartile25)
print("Quartile 50:",quartile50)
print("Quartile 75:",quartile75)


# NOVAS FEATURES: Tempo médio de resposta da polícia (Dispatch Time - Start Time)
crime["Dispatch Date / Time"] = pd.to_datetime(crime["Dispatch Date / Time"], format="%m/%d/%Y %I:%M:%S %p")
crime["Response_Time"] = (crime["Dispatch Date / Time"] - crime["Start_Date_Time"]).dt.total_seconds()

mean_response_time = crime["Response_Time"].mean()
print("Média do response time: ",mean_response_time)

median_response_time = crime["Response_Time"].median()
print("Mediana do response time: ",median_response_time)

variance_response_time = crime["Response_Time"].var()
print("Variância do response time: ",variance_response_time)

# Average police response time (dispatch time - start time) per police department
average_response_time = crime.groupby("Police District Name")["Response_Time"].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.fill_between(average_response_time["Police District Name"], average_response_time["Response_Time"], color='lightgreen', alpha=0.5)

plt.plot(average_response_time["Police District Name"], average_response_time["Response_Time"], color='blue', linewidth=2)

plt.xticks(rotation=45, ha='right')
plt.ylabel("Average Response Time in Seconds")
plt.title("Average Police Response Time per Department")

plt.tight_layout()
plt.show()


#POLICE DISTRICT NAME TO IDENTIFY CRITICAL AREAS
crime_counts = crime["Police District Name"].value_counts().reset_index()
crime_counts.columns = ['Police District Name', 'Crime Count']

plt.figure(figsize=(12, 6))
scatter = plt.scatter(x=crime_counts["Police District Name"],y=crime_counts["Crime Count"],c=crime_counts["Crime Count"],cmap="coolwarm", s=100, edgecolors='black')

plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Crimes")
plt.title("Crime Counts by Police District")

plt.colorbar(scatter, label="Number of Crimes")

plt.tight_layout()
plt.show()


#CREATE A HEAT MAP WITH LONGITUDE AND LATITUDE (TO VISUALIZE WHERE CRIMES ARE MOST FREQUENT)
m_1 = folium.Map(location=[39.1377,-77.13593], tiles="openstreetmap", zoom_start=10)
HeatMap(data=crime[["Latitude", "Longitude"]], radius=10).add_to(m_1)
m_1.save("crimeArea.html")


#----------------ERNESTAS-----------------------

# Pearson Correlations between Victims and Offence Code
encoder = LabelEncoder()
crime['Offence Code'] = encoder.fit_transform(crime['Crime Name1'])
correlation = crime[['Offence Code', 'Victims']].corr(method='pearson')
print("Pearson correlation between Victims and Offense Code:\n", correlation)


# Pearson Correlations between Victims and Crime Name 1
crime['CrimeCode'] = encoder.fit_transform(crime['Crime Name1'])
correlation_vict_crimename = crime[['CrimeCode', 'Victims']].corr(method='pearson')
print("Pearson correlation between Victims and Crime Name1:\n", correlation_vict_crimename)


# Pearson Correlations between Duration Crime and Crime Name1
crime["Start_Date_Time"] = pd.to_datetime(crime["Start_Date_Time"], format="%m/%d/%Y %I:%M:%S %p")
crime["End_Date_Time"] = pd.to_datetime(crime["End_Date_Time"], format="%m/%d/%Y %I:%M:%S %p")
crime["crime_duration"] = (crime["End_Date_Time"] - crime["Start_Date_Time"]).dt.total_seconds()

correlation_duration_crimename = crime[['crime_duration', 'CrimeCode']].corr(method='pearson')
print("Pearson correlation between Duration Crime and Crime Name1:\n", correlation_duration_crimename)

