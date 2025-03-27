import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap


crime = pd.read_csv("Crime.csv", low_memory=False)
crime = crime[crime["State"] == "MD"]

# Police District Name
ax = sns.countplot(x=crime["Police District Name"], data=crime, palette="Blues")
plt.ylabel("Crimes committed")
plt.show()

# Create a heat map with longitude and latitude (to visualize where crimes are most frequent)
m_1 = folium.Map(location=[39.1377,-77.13593], tiles="openstreetmap", zoom_start=10)
HeatMap(data=crime[["Latitude", "Longitude"]], radius=10).add_to(m_1)
m_1.save("crimeArea.html")


# Average police response time
crime["Start_Date_Time"] = pd.to_datetime(crime["Start_Date_Time"], format="%m/%d/%Y %I:%M:%S %p")
crime["Dispatch Date / Time"] = pd.to_datetime(crime["Dispatch Date / Time"], format="%m/%d/%Y %I:%M:%S %p")

crime["Response_Time"] = (crime["Dispatch Date / Time"] - crime["Start_Date_Time"]).dt.total_seconds()

# mean
mean_response_time = crime["Response_Time"].mean()
print(mean_response_time)

# median
median_response_time = crime["Response_Time"].median()
print(median_response_time)

#variance
variance_response_time = crime["Response_Time"].var()
print(variance_response_time)

# Crime duration
crime["End_Date_Time"] = pd.to_datetime(crime["End_Date_Time"], format="%m/%d/%Y %I:%M:%S %p")
crime["crime_duration"] = (crime["End_Date_Time"] - crime["Start_Date_Time"]).dt.total_seconds()

print(crime["crime_duration"])

#max
print("Max duration")
max_crime_duration = crime["crime_duration"].max()
print(max_crime_duration)

#min
print("Min duration")
min_crime_duration = crime["crime_duration"].min()
print(min_crime_duration)

#mean
print("Duration mean")
mean_crime_duration = crime["crime_duration"].mean()
print(mean_crime_duration)

#median
print("Duration median")
median_crime_duration = crime["crime_duration"].median()
print(median_crime_duration)

#var
print("Duration variance")
var_crime_duration = crime["crime_duration"].var()
print(var_crime_duration)

#quartiles
print("Duration quartiles")
quartile25 = crime["crime_duration"].quantile(0.25)
quartile50 = crime["crime_duration"].quantile(0.50)
quartile75 = crime["crime_duration"].quantile(0.75)
print(quartile25)
print(quartile50)
print(quartile75)



#average_response_time = crime.groupby("Police District Name")["Response_Time"].mean()
average_response_time = crime.groupby("Police District Name")["Response_Time"].mean().reset_index()
sns.barplot(x="Police District Name", y="Response_Time", data=average_response_time, palette="Blues")
plt.ylabel("Average Response Time in seconds")
#print(crime[(crime["Response_Time"] < 0) & (crime["Police District Name"] == "CITY OF TAKOMA PARK")])
#225434 linha
print(average_response_time)
plt.show()



