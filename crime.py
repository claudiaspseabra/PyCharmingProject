import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
import numpy as np
from wordcloud import WordCloud


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

# Crime Name 2
print("---------------------")
print("\n")
count_crime3 = crime["Crime Name2"].value_counts()
print(count_crime3)

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


# AVERAGE NUMBER OF VICTIMS PER CRIME
avg_victims_per_crime = crime.groupby("Crime Name1")["Victims"].mean().sort_values(ascending=False)
categories = avg_victims_per_crime.index
values = avg_victims_per_crime.values

N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
values_closed = np.append(values, values[0])

# Paleta Seaborn Deep
import seaborn as sns
pal_seaborn_deep = sns.color_palette("deep")[0]

# Criar o gráfico
fig, ax = plt.subplots(figsize=(8, 8), dpi=120, subplot_kw=dict(polar=True))
ax.fill(angles, values_closed, color=pal_seaborn_deep, alpha=0.25)
ax.plot(angles, values_closed, color=pal_seaborn_deep, linewidth=2)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10, ha='right')
ax.set_yticklabels([])

for i, value in enumerate(values):
    ax.text(angles[i], value + 0.1, f'{value:.2f}', ha='center', size=9, color=pal_seaborn_deep)

plt.title('Average Number of Victims per Crime', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# WHAT CRIMES HAVE THE MOST VICTIMS
top_crimes = crime.groupby("Crime Name1")["Victims"].sum().sort_values(ascending=False).head(10)

pal_seaborn_deep = sns.color_palette("deep")[1]
top_crimes.plot(kind='area', stacked=True, color=pal_seaborn_deep, alpha=0.7, title='Distribuição das Vítimas por Crime')

plt.ylabel('Número de Vítimas')
plt.show()


#----------------------------Joana---------------------

# Gráfico 1: Distribuição do Tipo de Crime
#crime_count = crime['Crime Name1'].value_counts()
#plt.figure(figsize=(10, 6))
#crime_count.plot(kind='barh', color=sns.color_palette("Set2", len(crime_count)))
#plt.title('Crime Type Distribution', fontsize=16)
#plt.xlabel('Number of Incidents', fontsize=12)
#plt.ylabel('Crime Type', fontsize=12)
#plt.tight_layout()
#plt.show()

# Gráfico 2: Crime Type Distribution for Each Police District (barras empilhadas
#crime_by_district = crime.groupby(['Police District Name', 'Crime Name1'])['Incident ID'].count().unstack()
#crime_by_district.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='Set3')
#plt.title('Crime Type Distribution for Each Police District', fontsize=16)
#plt.xlabel('Police District Name', fontsize=12)
#plt.ylabel('Number of Incidents', fontsize=12)
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()


# Gráfico 3: Most Committed Crime (Gráfico Nuvem)
deep_palette = sns.color_palette("deep")
colors_rgb = [tuple(int(c * 255) for c in deep_palette[i]) for i in range(1, 4)]

most_committed_crimes = crime['Crime Name1'].value_counts().head(3)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='dark_green',
    color_func=lambda word, **kwargs: colors_rgb[most_committed_crimes.index.get_loc(word)],
    relative_scaling=0.5
).generate_from_frequencies(most_committed_crimes.to_dict())

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Committed Crimes (Top 3)', fontsize=16)
plt.show()



#Distribution of Crime Type (each value in Crime Name1)
crime_counts = crime["Crime Name1"].value_counts()

plt.figure(figsize=(10, 6))
colors = sns.color_palette("deep")
plt.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Distribution of Crime Type")
plt.axis('equal')
plt.show()


#Crime Type Distribution for Each Police District
crime_pivot = crime.pivot_table(index="Police District Name", columns="Crime Name1", aggfunc="size", fill_value=0)

plt.figure(figsize=(12, 6))
sns.heatmap(crime_pivot, cmap="Purples", annot=False, linewidths=0.5)
plt.title("Crime Type Distribution for Each Police District")
plt.xlabel("Crime Type")
plt.ylabel("Police District")
plt.xticks(rotation=45)
plt.show()

#----------------JENIFER--------------------
#ESTE
# Number of Crimes by city
plt.figure(figsize=(12, 6))
sns.barplot(
    x=crime["City"].value_counts().index,
    y=crime["City"].value_counts().values,
    color=sns.color_palette("deep")[5],  # Usando apenas a sexta cor (índice 5)
    legend=False
)
plt.xlabel("City")
plt.ylabel("Number of Crimes")
plt.title("Number of Crimes by city")
plt.xticks(rotation=45, ha='right')
plt.show()



#OU ESTE (MOSTRA AS 5 CIDADES COM MAIS CRIMES, PARA FICAR MAIS LEGÍVEL)
# Top 5 Cities with Most Crimes
city_counts = crime["City"].value_counts()

top_5 = city_counts.head(5)
# Usando cores da paleta seaborn deep a partir da sétima cor
colors = sns.color_palette("deep")[6:11]  # Do índice 6 ao 10 (7ª à 11ª cor)

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    top_5.values,
    labels=top_5.index,
    autopct=lambda p: f'{int(p * sum(top_5.values) / 100)}',
    colors=colors,
    startangle=90,
    wedgeprops={'width': 0.4}
)

plt.title("Top 5 Cities with Most Crimes")
plt.show()


# Most Committed Crime by City
crime_city_top = crime.groupby('City')['Crime Name1'].agg(lambda x: x.value_counts().idxmax()).reset_index()

crime_counts = crime.groupby(['City', 'Crime Name1']).size().reset_index(name='Crime Count')

crime_city_top = crime_city_top.merge(crime_counts, on=['City', 'Crime Name1'])
crime_city_top = crime_city_top.sort_values(by='Crime Count', ascending=True)

plt.figure(figsize=(12, 8))
sns.barplot(y='City', x='Crime Count', hue='Crime Name1', data=crime_city_top, palette="deep")

plt.title('Most common crime by city', fontsize=16)
plt.xlabel('number of ocorrencies', fontsize=12)
plt.ylabel('City', fontsize=12)
plt.legend(title="Most common crime", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


# Crime by Patrol Area (using frequency Polygon)
crime_counts = crime["Beat"].value_counts()

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(
    x=crime_counts.index,
    y=crime_counts.values,
    marker="o",
    color=sns.color_palette("deep")[8],
    linewidth=2,
    markersize=8
)
plt.fill_between(crime_counts.index, crime_counts.values, color=sns.color_palette("deep")[8], alpha=0.2)

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
color = sns.color_palette("deep")[5]
average_response_time = crime.groupby("Police District Name")["Response_Time"].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.fill_between(average_response_time["Police District Name"], average_response_time["Response_Time"], color=color, alpha=0.5)

plt.plot(average_response_time["Police District Name"], average_response_time["Response_Time"], color=color, linewidth=2)

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



#------------EXPERIENCIAS-----------

#1. Crime Type Over The Years
crime["Year"] = crime["Start_Date_Time"].dt.year
crime_by_year = crime.groupby(["Year", "Crime Name1"])["Incident ID"].count().unstack()

crime_by_year.plot(kind='line', figsize=(12, 8), marker='o')
plt.title('Crime Type Distribution Over Time', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Incidents', fontsize=14)
plt.xticks(rotation=45)
plt.xticks(ticks=range(int(crime_by_year.index.min()), int(crime_by_year.index.max()) + 1))
plt.tight_layout()
plt.show()


#2. Crime Time of Day Distribution (gráfico de dispersão)
color = sns.color_palette("deep")[3]

crime["Hour"] = crime["Start_Date_Time"].dt.hour
crime_by_hour = crime.groupby("Hour")["Incident ID"].count()

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
sns.scatterplot(x=crime_by_hour.index, y=crime_by_hour, color=color, s=100, marker='D')

plt.title('Crime Incidents by Hour of the Day', fontsize=16, fontweight='bold')
plt.xlabel('Hour of the Day', fontsize=14)
plt.ylabel('Number of Crimes', fontsize=14)
plt.tight_layout()
plt.show()


#CRIME AGAINST PROPERTY A QUE HORAS OCORRE MAIS FREQUENTEMENTE
#crime["Hour"] = crime["Start_Date_Time"].dt.hour

#property_crimes = crime[crime["Crime Name1"] == "Crime Against Property"]

# Contar os crimes por hora
#property_crimes_by_hour = property_crimes.groupby("Hour")["Incident ID"].count()

# Gráfico de dispersão
#plt.figure(figsize=(12, 8))
#sns.set_style("whitegrid")
#sns.scatterplot(
 #   x=property_crimes_by_hour.index,
 #   y=property_crimes_by_hour.values,
 #   color='#E63946',  # podes trocar a cor se quiseres
 #   s=100,
 #   marker='D'
#)

#plt.title('Crime Against Property by Hour of the Day', fontsize=16, fontweight='bold')
#plt.xlabel('Hour of the Day', fontsize=14)
#plt.ylabel('Number of Crimes', fontsize=14)
#plt.tight_layout()
#plt.show()



color = sns.color_palette("deep")[7]
# 5. Comparing Response Time Across Different Crime Types
crime_response_time_by_type = crime.groupby('Crime Name1')['Response_Time'].mean().sort_values(ascending=False)

crime_response_time_by_type.plot(kind='barh', figsize=(12, 8), color=color)
plt.title('Average Police Response Time by Crime Type', fontsize=16)
plt.xlabel('Average Response Time (seconds)', fontsize=14)
plt.ylabel('Crime Type', fontsize=14)
plt.tight_layout()
plt.show()



#-----------CRIME NAMES-----------

# crime mais cometido (CrimeName2) por CrimeName1
top_main_types = crime["Crime Name1"].value_counts().head(3).index
filtered_crime = crime[crime["Crime Name1"].isin(top_main_types)]

grouped = filtered_crime.groupby(["Crime Name1", "Crime Name2"]).size().unstack(fill_value=0)

top_subtypes = grouped.sum(axis=0).sort_values(ascending=False).head(5).index
grouped = grouped[top_subtypes]

colors = sns.color_palette("deep", n_colors=len(top_subtypes))

grouped.plot(kind='bar', stacked=True, figsize=(12, 8), color=colors)

plt.title("Top 5 Subtypes (Crime Name2) in Top 3 Crime Categories (Crime Name1)", fontsize=14, fontweight='bold')
plt.xlabel("Main Crime Type (Crime Name1)")
plt.ylabel("Number of Crimes")
plt.legend(title="Subtype (Crime Name2)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


## crime mais cometido (crimeName3) por CrimeName2
top_subtypes2 = crime["Crime Name2"].value_counts().head(3).index
filtered_crime2 = crime[crime["Crime Name2"].isin(top_subtypes2)]

grouped2 = filtered_crime2.groupby(["Crime Name2", "Crime Name3"]).size().unstack(fill_value=0)

top_name3 = grouped2.sum(axis=0).sort_values(ascending=False).head(5).index
grouped2 = grouped2[top_name3]

colors2 = sns.color_palette("deep", n_colors=len(top_name3))

grouped2.plot(kind='bar', stacked=True, figsize=(12, 8), color=colors2)

plt.title("Top 5 Crime Name3 in Top 3 Crime Name2", fontsize=14, fontweight='bold')
plt.xlabel("Crime Subtype (Crime Name2)")
plt.ylabel("Number of Crimes")
plt.legend(title="Crime Detail (Crime Name3)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



# HORAS MAIS FREQUENTES DO CRIME NAME1
crime["Start_Date_Time"] = pd.to_datetime(crime["Start_Date_Time"])
crime["Hour"] = crime["Start_Date_Time"].dt.hour

def get_period(hour):
    if 6 <= hour < 12:
        return "Morning (6-11)"
    elif 12 <= hour < 18:
        return "Afternoon (12-17)"
    else:
        return "Night (18-5)"

crime["Period"] = crime["Hour"].apply(get_period)

top_main_types = crime["Crime Name1"].value_counts().head(3).index
filtered_crime = crime[crime["Crime Name1"].isin(top_main_types)]
crime_period_distribution = filtered_crime.groupby(["Crime Name1", "Period"]).size().unstack(fill_value=0)

period_order = ["Morning (6-11)", "Afternoon (12-17)", "Night (18-5)"]
crime_period_distribution = crime_period_distribution[period_order]

# Definindo cores para cada período
colors = {'Morning (6-11)': sns.color_palette("deep")[3],
          'Afternoon (12-17)': sns.color_palette("deep")[4],
          'Night (18-5)': sns.color_palette("deep")[5]}

color_list = [colors[period] for period in crime_period_distribution.columns]

plt.figure(figsize=(12, 8))
crime_period_distribution.plot(
    kind='bar',
    figsize=(12, 8),
    width=0.8,
    color=color_list,
    edgecolor='black'
)

plt.title("Crime Incidents by Time of Day for Each Crime Type", fontsize=14, fontweight='bold')
plt.xlabel("Crime Type")
plt.ylabel("Number of Crimes")
plt.legend(title="Time of Day", loc="upper left", bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()