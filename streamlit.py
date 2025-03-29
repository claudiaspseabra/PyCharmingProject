import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

sns.set(style='whitegrid')

@st.cache_data
def load_data():
    try:
        crime = pd.read_csv("Crime.csv")
        crime = crime[crime['State'] == "MD"]
    except Exception as e:
        st.error("Error loading dataset: " + str(e))
        return None
    return crime

crime = load_data()

# We need to choose between wide and centered
st.set_page_config(layout="centered", page_title="Crime Data Analysis", page_icon="📊")

st.title("Crime Data Analysis")

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("""
- [Dataset Description](#1-dataset-description)
- [Statistical Analysis](#2-statistical-analysis)
- [Advanced Feature Engineering](#3-advanced-feature-engineering)
- [Graphical Analysis](#4-graphical-analysis)
""")

# st.header("Project Requirements")
# st.markdown("""**???**""")

if crime is not None:

    st.header("1. Dataset Description")
    st.write("**Domain:** Crime Analysis")
    st.write("**Size:** {} rows and {} columns".format(crime.shape[0], crime.shape[1]))
    st.write("**Data Types:**")
    st.dataframe(crime.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Type'}))
    # st.write("Columns in the dataset:", crime.columns)
    # st.write("**Sample of First 5 Rows:**")
    # st.dataframe(crime.head())

    st.subheader("Feature Description Report")
    st.markdown("""
    **Incident ID:** A unique identifier for each crime incident.
    Serves as the primary key for tracking individual records.
    \n**Offence Code:** Code representing the specific offence committed, which helps classify and group incidents by offence type.
    \n**CR Number:** Unique crime report number.  
    \n**Dispatch Date / Time:** The date and time when the incident was dispatched.  
    \n**NIBRS Code:** Crime classification code based on the National Incident-Based Reporting System.  
    \n**Victims:** Number of victims involved in the incident.  
    \n**Crime Name1:** Primary category or name of the crime.  
    Serves as the main label for crime type.
    \n**Crime Name2:** Secondary crime category or descriptor. 
    Provides additional context for the incident.
    \n**Crime Name3:** Tertiary crime category or additional detail.  
    \n**Police District Name:** Name of the police district responsible for the incident.  
    \n**Block Address:** Block address where the incident occurred.  
    \n**City, State, Zip Code:** Location details of the incident.  
    \n**Agency:** Law enforcement agency involved.  
    \n**Place, Sector, Beat, PRA, Address Number, Street Prefix, Street Name, Street Suffix, Street Type:**  Detailed address and administrative information.  
    \n**Start_Date_Time & End_Date_Time:** Incident start and end times.
    \n**Latitude & Longitude:** Geographical coordinates of the incident.  
    \n**Police District Number, Location:** Additional location identifiers.  
    """)

    st.header("2. Statistical Analysis")

    st.write("**Modes**")
    modes = crime.mode().iloc[0]
    st.dataframe(modes.reset_index().rename(columns={'index': 'Column', 0: 'Mode'}))

    st.write("---")

    st.subheader("**Victims analysis**")

    # wrong?
    range_victims = crime['Victims'].max() - crime['Victims'].min()
    st.write("Range of victims: ", range_victims)

    # Victims Median
    median_victims = crime['Victims'].median()
    st.write("Median of Victims:", median_victims)

    # Victims Mean
    mean_victims = crime['Victims'].mean()
    st.write("Mean of Victims:", mean_victims)

    # Victims Max
    max_victims = crime['Victims'].max()
    st.write("Maximum number of Victims:", max_victims)

    # Victims Min
    min_victims = crime['Victims'].min()
    st.write("Minimum number of Victims:", min_victims)

    # Victims Quartiles
    quartiles_victims = crime['Victims'].quantile([0.25, 0.5, 0.75])
    st.write("Quartiles of Victims:")
    st.write(quartiles_victims)

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(crime['Victims'], kde=True, color='dodgerblue', bins=6)
    plt.title("Distribution of Victims")
    plt.xlim(0, 5)
    st.pyplot(fig)

    st.subheader("**CR Number Analysis**") # why?

    # CR Number Median
    median_cr_number = crime['CR Number'].median()
    st.write("\nMedian of CR Number:", median_cr_number)

    # CR Number Max
    max_cr_number = crime['CR Number'].max()
    st.write("Maximum number of CR Number:", max_cr_number)

    # CR Number Min
    min_cr_number = crime['CR Number'].min()
    st.write("Minimum number of CR Number:", min_cr_number)

    # -------------------------COUNTS--------------------------

    st.write("---")
    st.subheader("**Value counts in each column**")
    st.write("\n")

    # Crime Name 1 Value Count
    st.write("**Crime Name 1**")
    count_crime1 = crime['Crime Name1'].value_counts()
    st.write(count_crime1)

    # Crime Name 3 Value Count
    st.write("\n")
    st.write("**Crime Name 2**")
    count_crime3 = crime['Crime Name3'].value_counts()
    st.write(count_crime3)

    # Police District Name Value Count
    st.write("\n")
    st.write("**Police District Name**")
    count_police_district = crime['Police District Name'].value_counts()
    st.write(count_police_district)

    # City Value Count
    st.write("\n")
    st.write("**City**")
    count_city = crime['City'].value_counts()
    st.write(count_city)

    # Zip Code Value Count
    st.write("\n")
    st.write("**Zip Code**")
    count_zip_code = crime['Zip Code'].value_counts()
    st.write(count_zip_code)

    # Agency Value Count
    st.write("\n")
    st.write("**Agency**")
    count_agency = crime['Agency'].value_counts()
    st.write(count_agency)

    # Place Value Count
    st.write("\n")
    st.write("**Place**\n")
    count_place = crime['Place'].value_counts()
    st.write(count_place)

    # Street Type Value Count
    st.write("\n")
    st.write("**Street Type**")
    count_strtype = crime['Street Type'].value_counts()
    st.write(count_strtype)

    # -----------------VARIANCE-------------------

    st.write("---")
    st.write("**Variance**")

    # Offence Code
    crime['Offence Code Encoded'] = pd.factorize(crime['Offence Code'])[0]

    # Offence Code Variance
    variance_offence = crime['Offence Code Encoded'].var()
    st.write("Variance Offence Code: ", variance_offence)

    # Zip Code Variance
    variance_zip = crime['Zip Code'].var()
    st.write("Variance Zip Code: ", variance_zip)

    # Victims Variance
    variance_victims = crime['Victims'].var()
    st.write("Variance Victims: ", variance_victims)

    # --------------------COVARIANCE------------------------
    st.write("---")
    st.write("**Covariance**")

    crime['Police_District_code'] = pd.factorize(crime['Police District Name'])[0]
    crime['Crime_Name1_encoded'] = pd.factorize(crime['Crime Name1'])[0]
    crime['Zip_Code_code'] = pd.factorize(crime['Zip Code'])[0]
    crime['City_code'] = pd.factorize(crime['City'])[0]

    # Victims Covariance
    st.write("\n")
    st.write("*Victims Covariance*")

    covariance_victims_crime = crime['Victims'].cov(crime['Crime_Name1_encoded'])
    covariance_victims_police_district = crime['Victims'].cov(crime['Police_District_code'])
    covariance_victims_city = crime['Victims'].cov(crime['City_code'])
    covariance_victims_zip = crime['Victims'].cov(crime['Zip_Code_code'])

    st.write("Covariance between Victims and Crime Name 1:", covariance_victims_crime)
    st.write("Covariance between Victims and Police District Name:", covariance_victims_police_district)
    st.write("Covariance between Victims and City:", covariance_victims_city)
    st.write("Covariance between Victims and Zip Code:", covariance_victims_zip)

    # Zip Code Covariance
    st.write("\n")
    st.write("*Zip Code Covariance*")

    covariance_zip_crime = crime['Zip_Code_code'].cov(crime['Crime_Name1_encoded'])
    covariance_zip_police_district = crime['Zip_Code_code'].cov(crime['Police_District_code'])
    st.write("Covariance between Zip Code and Crime Name 1:", covariance_zip_crime)
    st.write("Covariance between Zip Code and Police District Name:", covariance_zip_police_district)

    # Crime Name 1 Covariance
    st.write("\n")
    st.write("*Crime Name 1 Covariance*")

    covariance_crime_police_district = crime['Crime_Name1_encoded'].cov(crime['Police_District_code'])
    covariance_crime_city = crime['Crime_Name1_encoded'].cov(crime['City_code'])
    covariance_crime_zip = crime['Crime_Name1_encoded'].cov(crime['Zip_Code_code'])

    st.write("Covariance between Crime Name 1 and Police District Name:", covariance_crime_police_district)
    st.write("Covariance between Crime Name 1 and City:", covariance_crime_city)
    st.write("Covariance between Crime Name 1 and Zip Code:", covariance_crime_zip)

    # --------------------CORRELATION------------------------
    st.write("---")
    st.write("**Correlation**")

    # Error - need to change and understand

    st.write("*Pearson Correlation between Victims and Offence Code*")
    correlation = crime[['Offence Code Encoded', 'Victims']].corr(method='pearson')
    st.write(correlation)

    st.write("*Pearson Correlation between Victims and Crime Name 1*")
    correlation_vict_crimename = crime[['Crime_Name1_encoded', 'Victims']].corr(method='pearson')
    st.write(correlation_vict_crimename)

    # not sure if this should stay here - crime duration is created ahead
    st.write("*Pearson Correlation between Duration Crime and Crime Name1*")
    crime['Start_Date_Time'] = pd.to_datetime(crime['Start_Date_Time'], format="%m/%d/%Y %I:%M:%S %p")
    crime['End_Date_Time'] = pd.to_datetime(crime['End_Date_Time'], format="%m/%d/%Y %I:%M:%S %p")
    crime['Crime Duration'] = (crime['End_Date_Time'] - crime['Start_Date_Time']).dt.total_seconds()

    correlation_duration_crimename = crime[['Crime Duration', 'Crime_Name1_encoded']].corr(method='pearson')
    st.write(correlation_duration_crimename)

    st.header("3. Advanced Feature Engineering")

    st.write("\n")
    st.write("**New feature: Crime Duration**")

    # New feature: Crime Duration
    crime['Start_Date_Time'] = pd.to_datetime(crime['Start_Date_Time'], format="%m/%d/%Y %I:%M:%S %p")
    crime['End_Date_Time'] = pd.to_datetime(crime['End_Date_Time'], format="%m/%d/%Y %I:%M:%S %p")
    crime['Crime Duration'] = (crime['End_Date_Time'] - crime['Start_Date_Time']).dt.total_seconds()

    # Crime Duration Mean
    mean_crime_duration = crime['Crime Duration'].mean()
    st.write("Crime Duration Mean: ", mean_crime_duration)

    # Crime Duration Median
    median_crime_duration = crime['Crime Duration'].median()
    st.write("Crime Duration Median: ", median_crime_duration)

    # Crime Duration Variance
    var_crime_duration = crime['Crime Duration'].var()
    st.write("Crime Duration Variance: ", var_crime_duration)

    # Crime Duration Min
    min_crime_duration = crime['Crime Duration'].min()
    st.write("Crime Duration Minimum: ", min_crime_duration)

    # Crime Duration Max
    max_crime_duration = crime['Crime Duration'].max()
    st.write("Crime Duration Maximum: ", max_crime_duration)

    st.write("Crime Duration Quartiles")
    quartile25 = crime['Crime Duration'].quantile(0.25)
    quartile50 = crime['Crime Duration'].quantile(0.50)
    quartile75 = crime['Crime Duration'].quantile(0.75)
    st.write("Quartile 25:", quartile25)
    st.write("Quartile 50:", quartile50)
    st.write("Quartile 75:", quartile75)

    st.write("\n")
    st.write("**New feature: Police Response Time**")

    # New Feature: Police Response Time (Dispatch Time - Start Time)
    crime['Dispatch Date / Time'] = pd.to_datetime(crime['Dispatch Date / Time'], format="%m/%d/%Y %I:%M:%S %p")
    crime['Response Time'] = (crime['Dispatch Date / Time'] - crime['Start_Date_Time']).dt.total_seconds()

    # Response Time Mean
    mean_response_time = crime['Response Time'].mean()
    st.write("Response Time Mean: ", mean_response_time)

    # Response Time Median
    median_response_time = crime['Response Time'].median()
    st.write("Response Time Median: ", median_response_time)

    # Response Time Variance
    variance_response_time = crime['Response Time'].var()
    st.write("Response Time Variance: ", variance_response_time)

    # ----------------------GRAPH ANALYSIS---------------------------------

    st.header("4. Graphical Analysis")

    crime['Year'] = crime['Start_Date_Time'].dt.year
    crime['Month'] = crime['Start_Date_Time'].dt.month

    col1, col2 = st.columns(2)

    with col1:
        fig = plt.figure(figsize=(10, 6))
        crimeByYear = sns.countplot(data=crime, x='Year')
        crimeByYear.set(ylabel='Crimes')
        plt.title("Yearly Crime Distribution")
        st.pyplot(fig)

    with col2:
        fig = plt.figure(figsize=(10, 6))
        crimeByMonth = sns.countplot(data=crime, x='Month')
        crimeByMonth.set(ylabel='Crimes')
        plt.title("Monthly Crime Distribution")
        st.pyplot(fig)

    # # 1. Yearly crime distribution
    # fig = plt.figure(figsize=(12, 10))
    # crimeByYear = sns.countplot(data=crime, x='Year')
    # crimeByYear.set(ylabel='Crimes')
    # plt.title("Yearly crime distribution")
    # st.pyplot(fig)
    #
    # # 2. Monthly crime distribution
    # fig = plt.figure(figsize=(12, 10))
    # crimeByMonth = sns.countplot(data=crime, x='Month')
    # crimeByMonth.set(ylabel='Crimes')
    # plt.title("Monthly crime distribution")
    # st.pyplot(fig)

    fig = plt.figure(figsize=(10, 6))
    crimeByMonth = sns.countplot(data=crime, x='Month')
    crimeByMonth.set(ylabel='Crimes')
    plt.title("Monthly Crime Distribution")
    st.pyplot(fig)

    # 3. Monthly crime distribution by year
    # crimeByMonthYear = [crime[crime['Year'] == year]['Month'].value_counts() for year in list(crime['Year'].unique())]
    # i = 0
    # for year in list(crime['Year'].unique()):
    #     year = 2016 + i
    #     crimeYear = crime[crime['Year'] == year]
    #
    #     fig = plt.figure(figsize=(12, 10))
    #
    #     crimeMonthYear = sns.countplot(data=crimeYear, x='Month')
    #     crimeMonthYear.set(ylabel='Crimes')
    #     plt.title(f"Monthly crime in {year}")
    #
    #     st.pyplot(fig)
    #
    #     i += 1

    crimeByMonthYear = [crime[crime['Year'] == year]['Month'].value_counts() for year in list(crime['Year'].unique())]

    i = 0
    for j in range(0, len(crime['Year'].unique()), 2):
        col1, col2 = st.columns(2)
        with col1:
            year = 2016 + j
            crimeYear = crime[crime['Year'] == year]

            fig = plt.figure(figsize=(12, 10))
            crimeMonthYear = sns.countplot(data=crimeYear, x='Month')
            crimeMonthYear.set(ylabel='Crimes')
            plt.title(f"Monthly crime in {year}")
            st.pyplot(fig)

        with col2:
            if j + 1 < len(crime['Year'].unique()):
                year = 2016 + j + 1
                crimeYear = crime[crime['Year'] == year]

                fig = plt.figure(figsize=(12, 10))
                crimeMonthYear = sns.countplot(data=crimeYear, x='Month')
                crimeMonthYear.set(ylabel='Crimes')
                plt.title(f"Monthly crime in {year}")
                st.pyplot(fig)

    # 4. Distribution of Crime Type
    crime_count = crime['Crime Name1'].value_counts()
    fig = plt.figure(figsize=(10, 6))
    crime_count.plot(kind='barh', color=sns.color_palette("Set2", len(crime_count)))
    plt.title("Crime Type Distribution", fontsize=16)
    plt.xlabel("Number of Incidents", fontsize=12)
    plt.ylabel("Crime Type", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    # 4. Distribution of Crime Type (each value in Crime Name1)
    crime_counts = crime['Crime Name1'].value_counts()

    fig = plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    plt.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title("Distribution of Crime Type")
    plt.axis('equal')
    st.pyplot(fig)

    # 5. Crime Type Distribution for Each Police District
    crime_pivot = crime.pivot_table(index="Police District Name", columns='Crime Name1', aggfunc='size', fill_value=0)

    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(crime_pivot, cmap='Blues', annot=False, linewidths=0.5)
    plt.title("Crime Type Distribution for Each Police District")
    plt.xlabel("Crime Type")
    plt.ylabel("Police District")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # this one doesn't show up for some reason

    # 5. Crime Type Distribution for Each Police District (barras empilhadas)
    crime_by_district = crime.groupby(['Police District Name', 'Crime Name1'])['Incident ID'].count().unstack()
    # fig = plt.figure(figsize=(12, 8))

    crime_by_district.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='Set3')
    fig = plt.figure(figsize=(12, 8))

    plt.title("Crime Type Distribution for Each Police District")
    plt.xlabel("Police District Name")
    plt.ylabel("Number of Incidents")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # estranho ???
    # Most Committed Crime (barra vertical)
    most_committed_crime = crime['Crime Name1'].value_counts().head(1)
    fig = plt.figure(figsize=(8, 6))
    most_committed_crime.plot(kind='bar', color='tomato')
    plt.title("Most Committed Crime", fontsize=16)
    plt.xlabel("Crime Type", fontsize=12)
    plt.ylabel("Number of Incidents", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    # wrong?
    # 6. Most Committed Crime by City
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(
        x=crime['City'].value_counts().index,
        y=crime['City'].value_counts().values,
        hue=crime['City'].value_counts().index,
        palette='coolwarm',
        legend=False
    )
    plt.xlabel("City")
    plt.ylabel("Number of Crimes")
    plt.title("Most Committed Crime by City")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # 7. Crime by Patrol Area (using frequency Polygon)
    crime_counts = crime['Beat'].value_counts()

    sns.set(style='whitegrid')
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(
        x=crime_counts.index,
        y=crime_counts.values,
        marker='o',
        color='purple',
        linewidth=2,
        markersize=8
    )
    plt.fill_between(crime_counts.index, crime_counts.values, color='purple', alpha=0.2)

    plt.title("Crime by Patrol Area")
    plt.xlabel("Patrol Area (Beat)")
    plt.ylabel("Number of Crimes")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # 8. Crime by ZIP Code
    crime_by_zip = crime.groupby('Zip Code')['Incident ID'].count()

    fig = plt.figure(figsize=(12, 8))
    plt.scatter(crime_by_zip.index, crime_by_zip.values, c=crime_by_zip.values, cmap='coolwarm', s=100)

    plt.title("Crime by Zip Code", fontsize=16, fontweight='bold')
    plt.xlabel("Zip Code", fontsize=14)
    plt.ylabel("Number of Crimes", fontsize=14)

    plt.colorbar(label="Number of Crimes")
    st.pyplot(fig)

    # 9. Average number of victims per crime
    avg_victims_per_crime = crime.groupby('Crime Name1')['Victims'].mean().sort_values(ascending=False)
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

    # Adding the average number of victims
    for i, value in enumerate(values):
        ax.text(angles[i], value + 0.1, f'{value:.2f}', horizontalalignment='center', size=10, color='dodgerblue',
                weight='semibold')

    plt.title("Average Number of Victims per Crime", fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    # 10. What crimes have the most victims
    fig = plt.figure(figsize=(12, 8))
    top_crimes = crime.groupby('Crime Name1')['Victims'].sum().sort_values(ascending=False).head(10)
    top_crimes.plot(kind='area', stacked=True, color='skyblue', alpha=0.7, title='Victims Distribution per Crime')

    plt.ylabel("Number of victims")
    st.pyplot(fig)

    # 11. Average police response time (dispatch time - start time) per police department
    average_response_time = crime.groupby('Police District Name')['Response Time'].mean().reset_index()

    fig = plt.figure(figsize=(12, 6))
    plt.fill_between(average_response_time['Police District Name'], average_response_time['Response Time'],
                     color='lightgreen', alpha=0.5)

    plt.plot(average_response_time['Police District Name'], average_response_time['Response Time'], color='blue',
             linewidth=2)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Average Response Time in Seconds")
    plt.title("Average Police Response Time per Department")

    plt.tight_layout()
    st.pyplot(fig)

    # 12. Crime count by police district name
    crime_counts = crime['Police District Name'].value_counts().reset_index()
    crime_counts.columns = ['Police District Name', 'Crime Count']

    fig = plt.figure(figsize=(12, 6))
    scatter = plt.scatter(x=crime_counts['Police District Name'], y=crime_counts['Crime Count'],
                          c=crime_counts['Crime Count'], cmap='coolwarm', s=100, edgecolors='black')

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of Crimes")
    plt.title("Crime Counts by Police District")

    plt.colorbar(scatter, label='Number of Crimes')

    plt.tight_layout()
    st.pyplot(fig)

    # 13. Heat map with longitude and latitude
    m_1 = folium.Map(location=[39.1377, -77.13593], tiles="openstreetmap", zoom_start=10)
    HeatMap(data=crime[["Latitude", "Longitude"]], radius=10).add_to(m_1)
    st_folium(m_1, width=700)

else:
    st.write("Failed to load the dataset. Please ensure that 'Crime.csv' is in the working directory.")

if st.button("Back to Top"):
    st.experimental_rerun()