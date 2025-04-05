import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap


crime = pd.read_csv("Crime.csv", low_memory=False)
crime = crime[crime["State"] == "MD"]

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