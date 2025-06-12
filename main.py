import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from wordcloud import WordCloud
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import io
from fpdf import FPDF

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time
from sklearn.decomposition import PCA

# Wide or centered?
st.set_page_config(layout="centered", page_title="Crime Data Analysis", page_icon="📊")

st.title("Crime Data Analysis")
sns.set(style='whitegrid')


def save_plot_to_image(figure):
    img_buffer = io.BytesIO()
    figure.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    return img_buffer


def generate_pdf(figures):
    pdf = FPDF()

    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(200, 10, txt="Crime graphics", ln=True, align="C")

    for i, figure in enumerate(figures):
        if i > 0:
            pdf.add_page()

        img_buffer = save_plot_to_image(figure)
        img_path = f"images/plot_image_{i}.png"
        with open(img_path, 'wb') as f:
            f.write(img_buffer.read())

        pdf.image(img_path, x=10, y=30, w=180)

    pdf_output_path = "pdfs/crime_graph.pdf"
    pdf.output(pdf_output_path)

    return pdf_output_path


@st.cache_data
def load_data():
    try:
        crime = pd.read_csv("Crime.csv", low_memory=False)
        crime = crime[crime['State'] == "MD"].copy()
        crime['Start_Date_Time'] = pd.to_datetime(crime['Start_Date_Time'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
        crime['End_Date_Time'] = pd.to_datetime(crime['End_Date_Time'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
        crime['Dispatch Date / Time'] = pd.to_datetime(crime['Dispatch Date / Time'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
        crime['Crime Duration'] = (crime['End_Date_Time'] - crime['Start_Date_Time']).dt.total_seconds()
        crime['Response Time'] = (crime['Dispatch Date / Time'] - crime['Start_Date_Time']).dt.total_seconds()
        return crime
    except Exception as e:
        st.error("Error loading dataset: " + str(e))
        return None

crime = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
selected_analysis = st.sidebar.radio(
    "Choose an analysis:",
    ["DataSet Description", "Advanced Feature Engineering", "Statistical Analysis", "Graphical Analysis", "Prediction"]
)

if crime is not None:
    if selected_analysis == "DataSet Description":
        st.header("1. Dataset Description")
        st.write("This dataset, published on Kaggle by Tarun Tirawi, contains data about "
                 "crimes committed in the state of Maryland, in the United States of America, "
                 "between July 2016 and December 2020.")
        st.write("**Domain:** Crime Analysis")
        st.write("**Size:** {} rows and {} columns".format(crime.shape[0], crime.shape[1]))
        st.write("**Data Types:**")
        st.dataframe(crime.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Type'}))
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
        \n**Place, Sector, Beat, PRA, Address Number, Street Prefix, Street Name, Street Suffix, Street Type:** Detailed address and administrative information.
        \n**Start_Date_Time & End_Date_Time:** Incident start and end times.
        \n**Latitude & Longitude:** Geographical coordinates of the incident.
        \n**Police District Number, Location:** Additional location identifiers.
        """)

    elif selected_analysis == "Advanced Feature Engineering":
        st.header("2. Advanced Feature Engineering")

        st.write("\n")
        st.subheader("**New feature: Crime Duration**")
        st.write("*Crime duration is the time difference between the Start_Date_Time and End_Date_Time.*")

        crime['Crime Duration'] = (crime['End_Date_Time'] - crime['Start_Date_Time']).dt.total_seconds()

        st.write("\n")
        st.write("Crime Duration Mean: ", crime['Crime Duration'].mean())
        st.write("Crime Duration Median: ", crime['Crime Duration'].median())
        st.write("Crime Duration Variance: ", crime['Crime Duration'].var())
        st.write("Crime Duration Minimum: ", crime['Crime Duration'].min())
        st.write("Crime Duration Maximum: ", crime['Crime Duration'].max())
        st.write("Crime Duration Quartiles")
        st.write("Quartile 25:", crime['Crime Duration'].quantile(0.25))
        st.write("Quartile 50:", crime['Crime Duration'].quantile(0.50))
        st.write("Quartile 75:", crime['Crime Duration'].quantile(0.75))

        # # Gráficos para Crime Duration (sugestão)
        # st.write("**Crime Duration Analysis**")
        # fig_duration_hist = plt.figure(figsize=(8, 4))
        # sns.histplot(crime['Crime Duration'].dropna(), kde=True)
        # plt.title('Distribution of Crime Duration')
        # plt.xlabel('Duration (seconds)')
        # plt.ylabel('Frequency')
        # st.pyplot(fig_duration_hist)
        #
        # fig_duration_boxplot = plt.figure(figsize=(8, 4))
        # sns.boxplot(x=crime['Crime Duration'].dropna())
        # plt.title('Boxplot of Crime Duration')
        # plt.xlabel('Duration (seconds)')
        # st.pyplot(fig_duration_boxplot)

        st.write("\n")
        st.subheader("**New feature: Police Response Time**")
        st.write("*Response Time is the time difference between the Start_Date_Time and Dispatch Date/Time.*")

        crime['Response Time'] = (crime['Dispatch Date / Time'] - crime['Start_Date_Time']).dt.total_seconds()

        st.write("\n")
        st.write("Response Time Mean: ", crime['Response Time'].mean())
        st.write("Response Time Median: ", crime['Response Time'].median())
        st.write("Response Time Variance: ", crime['Response Time'].var())

        # st.write("\n")
        # st.write("**Police Response Time Analysis**")
        # # GRAPH: Average Police Response Time per Police Department
        # average_response_time = crime.groupby('Police District Name')['Response Time'].mean().reset_index()
        # color = sns.color_palette("deep")[5]
        # fig_resp_dept = plt.figure(figsize=(12, 6))
        # plt.fill_between(average_response_time['Police District Name'], average_response_time['Response Time'], color=color, alpha=0.5)
        # plt.plot(average_response_time['Police District Name'], average_response_time['Response Time'], color=color, linewidth=2)
        # plt.xticks(rotation=45, ha='right')
        # plt.ylabel("Average Response Time in Seconds")
        # plt.title("Average Police Response Time per Department")
        # plt.tight_layout()
        # st.pyplot(fig_resp_dept)
        #
        # # GRAPH: Average Police Response Time by Crime Type
        # crime_response_time_by_type = crime.groupby('Crime Name1')['Response Time'].mean().sort_values(ascending=False)
        # fig_resp_crime = plt.figure(figsize=(12, 6))
        # crime_response_time_by_type.plot(kind='barh', color=color)
        # plt.title('Average Police Response Time by Crime Type', fontsize=16)
        # plt.xlabel('Average Response Time (seconds)', fontsize=14)
        # plt.ylabel('Crime Type', fontsize=14)
        # plt.tight_layout()
        # st.pyplot(fig_resp_crime)

    elif selected_analysis == "Statistical Analysis":
        st.header("3. Statistical Analysis")

        st.write("**Modes**")
        modes = crime.mode().iloc[0]
        st.dataframe(modes.reset_index().rename(columns={'index': 'Column', 0: 'Mode'}))

        st.write("---")
        st.subheader("**Value counts in each column**")
        st.write("\n")
        st.write("**Crime Name 1**")
        st.write(crime['Crime Name1'].value_counts())
        st.write("\n")
        st.write("**Crime Name 2**")
        st.write(crime['Crime Name2'].value_counts())
        st.write("\n")
        st.write("**Police District Name**")
        st.write(crime['Police District Name'].value_counts())
        st.write("\n")
        st.write("**City**")
        st.write(crime['City'].value_counts())
        st.write("\n")
        st.write("**Zip Code**")
        st.write(crime['Zip Code'].value_counts())
        st.write("\n")
        st.write("**Agency**")
        st.write(crime['Agency'].value_counts())
        st.write("\n")
        st.write("**Place**\n")
        st.write(crime['Place'].value_counts())
        st.write("\n")
        st.write("**Street Type**")
        st.write(crime['Street Type'].value_counts())

        st.write("---")
        st.write("**Variance**")
        crime['Offence Code Encoded'] = pd.factorize(crime['Offence Code'])[0]
        st.write("Variance Offence Code: ", crime['Offence Code Encoded'].var())
        st.write("Variance Zip Code: ", crime['Zip Code'].var())
        st.write("Variance Victims: ", crime['Victims'].var())

        st.write("---")
        st.write("**Covariance**")
        crime['Police District Encoded'] = pd.factorize(crime['Police District Name'])[0]
        crime['Crime Name1 Encoded'] = pd.factorize(crime['Crime Name1'])[0]
        crime['Zip Code Encoded'] = pd.factorize(crime['Zip Code'])[0]
        crime['City Encoded'] = pd.factorize(crime['City'])[0]

        st.write("\n")
        st.write("*Victims Covariance*")
        st.write("Covariance between Victims and Crime Name 1:", crime['Victims'].cov(crime['Crime Name1 Encoded']))
        st.write("Covariance between Victims and Police District Name:", crime['Victims'].cov(crime['Police District Encoded']))
        st.write("Covariance between Victims and City:", crime['Victims'].cov(crime['City Encoded']))
        st.write("Covariance between Victims and Zip Code:", crime['Victims'].cov(crime['Zip Code Encoded']))

        st.write("\n")
        st.write("*Zip Code Covariance*")
        st.write("Covariance between Zip Code and Crime Name 1:", crime['Zip Code Encoded'].cov(crime['Crime Name1 Encoded']))
        st.write("Covariance between Zip Code and Police District Name:", crime['Zip Code Encoded'].cov(crime['Police District Encoded']))

        st.write("\n")
        st.write("*Crime Name 1 Covariance*")
        st.write("Covariance between Crime Name 1 and Police District Name:", crime['Crime Name1 Encoded'].cov(crime['Police District Encoded']))
        st.write("Covariance between Crime Name 1 and City:", crime['Crime Name1 Encoded'].cov(crime['City Encoded']))

        st.write("---")
        st.write("**Correlation**")
        st.write("*Pearson Correlation between Victims and Offence Code*")
        st.write(crime[['Offence Code Encoded', 'Victims']].corr(method='pearson'))
        st.write("*Pearson Correlation between Victims and Crime Name 1*")
        st.write(crime[['Crime Name1 Encoded', 'Victims']].corr(method='pearson'))
        st.write("*Pearson Correlation between Duration Crime and Crime Name1*")
        crime['Crime Duration'] = (crime['End_Date_Time'] - crime['Start_Date_Time']).dt.total_seconds()
        st.write(crime[['Crime Duration', 'Crime Name1 Encoded']].corr(method='pearson'))

        st.write("---")
        st.subheader("**Victims analysis**")
        st.write("Maximum number of Victims:", crime['Victims'].max())
        st.write("Minimum number of Victims:", crime['Victims'].min())
        st.write("Range of victims: ", crime['Victims'].max() - crime['Victims'].min())
        st.write("Median of Victims:", crime['Victims'].median())
        st.write("Mean of Victims:", crime['Victims'].mean())
        st.write("Quartiles of Victims:")
        st.write(crime['Victims'].quantile([0.25, 0.5, 0.75]))


    #------------GRÁFICOS-----------

    elif selected_analysis == "Graphical Analysis":
        st.header("4. Graphical Analysis")

        analysis_type = st.radio(
            "Choose a graphical analysis:",
            [
                "Crime Analysis",
                "Analysis of Crime Occurrence and Time",
                "Analysis of Crime Location",
                "Analysis of Police Response Time",
                "Analysis of Crime Victims",
            ]
        )

        colors = plt.cm.tab10.colors
        figures = []

        if analysis_type == "Crime Analysis":
            st.subheader("Crime Analysis")
            st.write("\n")

            # 2. Distribution of Crime Type in %
            crime_counts = crime['Crime Name1'].value_counts()
            fig2 = plt.figure(figsize=(10, 6))
            plt.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
            plt.title("Distribution of Crime Type")
            plt.axis('equal')
            st.pyplot(fig2)
            figures.append(fig2)

            st.write("---")

            # 3. Most Committed Crime Types
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

            fig3 = plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Committed Crimes (Top 3)', fontsize=16)
            st.pyplot(fig3)
            figures.append(fig3)

            st.write("---")

            # 4. Most committed crime (CrimeName2) per CrimeName1
            top_main_types = crime["Crime Name1"].value_counts().head(3).index
            filtered_crime = crime[crime["Crime Name1"].isin(top_main_types)]

            grouped = filtered_crime.groupby(["Crime Name1", "Crime Name2"]).size().unstack(fill_value=0)
            top_subtypes = grouped.sum(axis=0).sort_values(ascending=False).head(5).index
            grouped = grouped[top_subtypes]

            colors = sns.color_palette("deep", n_colors=len(top_subtypes))

            fig4, ax = plt.subplots(figsize=(12, 8))
            grouped.plot(kind='bar', stacked=True, figsize=(12, 8), color=colors, ax=ax)

            ax.set_title("Top 5 Subtypes in Top 3 Crime Categories", fontsize=14, fontweight='bold')
            ax.set_xlabel("Main Crime Type (Crime Name1)")
            ax.set_ylabel("Number of Crimes")
            ax.legend(title="Subtype (Crime Name2)")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig4)
            figures.append(fig4)

            st.write("---")

            # 5. Most committed crime (CrimeName3) per CrimeName2
            top_subtypes2 = crime["Crime Name2"].value_counts().head(3).index
            filtered_crime2 = crime[crime["Crime Name2"].isin(top_subtypes2)]

            grouped2 = filtered_crime2.groupby(["Crime Name2", "Crime Name3"]).size().unstack(fill_value=0)

            top_name3 = grouped2.sum(axis=0).sort_values(ascending=False).head(5).index
            grouped2 = grouped2[top_name3]

            colors2 = sns.color_palette("deep", n_colors=len(top_name3))

            fig5, ax2 = plt.subplots(figsize=(12, 8))
            grouped2.plot(kind='bar', stacked=True, figsize=(12, 8), color=colors2, ax=ax2)

            ax2.set_title("Top 5 Crime Name3 in Top 3 Crime Name2", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Crime Subtype (Crime Name2)")
            ax2.set_ylabel("Number of Crimes")
            ax2.legend(title="Crime Detail (Crime Name3)")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig5)
            figures.append(fig5)
            #st.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")

            col1, col2, col3 = st.columns([7.5, 5, 4])

            with col2:
                if st.button("Create PDF"):
                    pdf_path = generate_pdf(figures)
                    st.success("PDF created with success!")

                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF", f, file_name="crime_report.pdf", mime="application/pdf")


        elif analysis_type == "Analysis of Crime Occurrence and Time":
            st.subheader("Analysis of Crime Occurrence and Time")
            st.write("---")

            crime['Year'] = crime['Start_Date_Time'].dt.year

            crime['Month'] = crime['Start_Date_Time'].dt.month

            col1, col2 = st.columns(2)

            with col1:
                # 6. Yearly Crime Distribution
                fig6 = plt.figure(figsize=(10, 6))

                crimeByYear = sns.countplot(data=crime, x='Year')

                crimeByYear.set(ylabel='Crimes')
                plt.title("Yearly Crime Distribution")
                st.pyplot(fig6)
                figures.append(fig6)

            with col2:
                # 7. Monthly Crime Distribution (Overall)
                fig7 = plt.figure(figsize=(10, 6))

                crimeByMonth = sns.countplot(data=crime, x='Month')
                crimeByMonth.set(ylabel='Crimes')
                plt.title("Monthly Crime Distribution (Overall)")
                st.pyplot(fig7)
                figures.append(fig7)

            st.write("---")

            # 9. Crime Type Distribution Over Time
            crime["Year"] = crime["Start_Date_Time"].dt.year

            crime_by_year = crime.groupby(["Year", "Crime Name1"])["Incident ID"].count().unstack(fill_value=0)

            fig11 = plt.figure(figsize=(12, 6))
            crime_by_year.plot(kind='line', marker='o', ax=fig11.gca())

            plt.title('Crime Type Distribution Over Time', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Number of Incidents', fontsize=14)
            plt.xticks(rotation=45)

            plt.xticks(ticks=range(int(crime_by_year.index.min()), int(crime_by_year.index.max()) + 1))
            plt.tight_layout()
            st.pyplot(fig11)
            figures.append(fig11)

            st.write("---")

            # 10. Crime Incidents by Hour of the Day
            color_hour = sns.color_palette("deep")[3]
            crime["Hour"] = crime["Start_Date_Time"].dt.hour
            crime_by_hour = crime.groupby("Hour")["Incident ID"].count()

            fig12 = plt.figure(figsize=(10, 5))
            sns.set_style("whitegrid")
            sns.scatterplot(x=crime_by_hour.index, y=crime_by_hour, color=color_hour, s=100, marker='D')

            plt.title('Crime Incidents by Hour of the Day', fontsize=14, fontweight='bold')
            plt.xlabel('Hour of the Day', fontsize=12)
            plt.ylabel('Number of Crimes', fontsize=12)
            plt.tight_layout()

            st.pyplot(fig12)
            figures.append(fig12)

            st.write("---")

            # 11. Most Frequent Hours of Crime Name1
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
            filtered_crime_period = crime[crime["Crime Name1"].isin(top_main_types)]

            crime_period_distribution = filtered_crime_period.groupby(["Crime Name1", "Period"]).size().unstack(fill_value=0)

            period_order = ["Morning (6-11)", "Afternoon (12-17)", "Night (18-5)"]
            crime_period_distribution = crime_period_distribution.reindex(columns=period_order, fill_value=0)

            colors_period = {'Morning (6-11)': sns.color_palette("deep")[3],
                             'Afternoon (12-17)': sns.color_palette("deep")[4],
                             'Night (18-5)': sns.color_palette("deep")[5]}

            color_list_period = [colors_period[period] for period in crime_period_distribution.columns]
            fig13 = plt.figure(figsize=(10, 6))

            crime_period_distribution.plot(
                kind='bar',
                figsize=(10, 6),
                width=0.8,
                color=color_list_period,
                edgecolor='black',
                ax=fig13.gca()
            )

            plt.title("Crime Incidents by Time of Day for Each Crime Type", fontsize=14, fontweight='bold')
            plt.xlabel("Crime Type")
            plt.ylabel("Number of Crimes")
            plt.legend(title="Time of Day", loc="upper right")
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()
            st.pyplot(fig13)
            figures.append(fig13)
            st.write("\n")
            st.write("\n")
            st.write("\n")

            col1, col2, col3 = st.columns([7.5, 5, 4])

            with col2:
                if st.button("Create PDF"):
                    pdf_path = generate_pdf(figures)
                    st.success("PDF created with success!")

                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF", f, file_name="crime_report.pdf", mime="application/pdf")

        elif analysis_type == "Analysis of Crime Location":
            st.subheader("Analysis of Crime Location")
            st.write("\n")

            # 12. Number of Crimes by city
            fig14 = plt.figure(figsize=(10, 6))
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
            st.pyplot(fig14)
            figures.append(fig14)

            st.write("---")

            # 13. Top 5 Cities with Most Crimes
            city_counts = crime["City"].value_counts()

            top_5 = city_counts.head(5)
            colors_top_cities = sns.color_palette("deep")[6:11]

            fig15 = plt.figure(figsize=(10, 6))
            wedges, texts, autotexts = plt.pie(
                top_5.values,
                labels=top_5.index,
                autopct=lambda p: f'{int(p * sum(top_5.values) / 100)}',
                colors=colors_top_cities,
                startangle=90,
                wedgeprops={'width': 0.4}
            )

            plt.title("Top 5 Cities with Most Crimes")
            st.pyplot(fig15)
            figures.append(fig15)

            st.write("---")

            # 14. Most Committed Crime by City
            crime_city_top = crime.groupby('City')['Crime Name1'].agg(lambda x: x.value_counts().idxmax()).reset_index()
            crime_counts_city = crime.groupby(['City', 'Crime Name1']).size().reset_index(name='Crime Count')

            crime_city_top = crime_city_top.merge(crime_counts_city, on=['City', 'Crime Name1'])
            crime_city_top = crime_city_top.sort_values(by='Crime Count', ascending=True)

            fig16 = plt.figure(figsize=(10, 6))
            sns.barplot(y='City', x='Crime Count', hue='Crime Name1', data=crime_city_top, palette="deep")

            plt.title('Most Common Crime by City', fontsize=16)
            plt.xlabel('Number of Occurrences', fontsize=12)
            plt.ylabel('City', fontsize=12)
            plt.legend(title="Most common crime", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='x', linestyle='--', alpha=0.6)

            st.pyplot(fig16)
            figures.append(fig16)

            st.write("---")

            # 15. Crime by Patrol Area
            crime_counts_beat = crime['Beat'].value_counts()

            sns.set(style='whitegrid')
            fig17 = plt.figure(figsize=(12, 6))
            sns.lineplot(
                x=crime_counts_beat.index,
                y=crime_counts_beat.values,
                marker='o',
                color=sns.color_palette("deep")[8],
                linewidth=2,
                markersize=8
            )
            plt.fill_between(crime_counts_beat.index, crime_counts_beat.values, color=sns.color_palette("deep")[8],
                             alpha=0.2)

            plt.title("Crime by Patrol Area")
            plt.xlabel("Patrol Area (Beat)")
            plt.ylabel("Number of Crimes")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig17)
            figures.append(fig17)

            st.write("---")

            # 16. Crime by ZIP Code
            crime_by_zip = crime.groupby('Zip Code')['Incident ID'].count()
            crime_by_zip_filtered = crime_by_zip[(crime_by_zip.index >= 20588) & (crime_by_zip.index <= 21930)]

            fig18 = plt.figure(figsize=(12, 8))
            plt.scatter(crime_by_zip_filtered.index, crime_by_zip_filtered.values, c=crime_by_zip_filtered.values,
                        cmap='coolwarm',
                        s=100)

            plt.title("Crime by Zip Code", fontsize=16, fontweight='bold')
            plt.xlabel("Zip Code", fontsize=14)
            plt.ylabel("Number of Crimes", fontsize=14)

            plt.colorbar(label="Number of Crimes")
            st.pyplot(fig18)

            figures.append(fig18)

            st.write("---")

            # 17. Crime Type Distribution for Each Police District
            crime_pivot = crime.pivot_table(index="Police District Name", columns='Crime Name1', aggfunc='size',
                                            fill_value=0)

            fig19 = plt.figure(figsize=(10, 6))
            sns.heatmap(crime_pivot, cmap='Purples', annot=False, linewidths=0.5)
            plt.title("Crime Type Distribution for Each Police District")
            plt.xlabel("Crime Type")
            plt.ylabel("Police District")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig19)
            figures.append(fig19)

            st.write("---")

            # 18. Crime Type Distribution for Each Police District (Bar Chart)
            crime_by_district = crime.groupby(['Police District Name', 'Crime Name1'])['Incident ID'].count().unstack()
            fig20 = plt.figure(figsize=(12, 8))
            crime_by_district.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='Set3', ax=fig20.gca())
            plt.title("Crime Type Distribution for Each Police District")
            plt.xlabel("Police District Name")
            plt.ylabel("Number of Incidents")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig20)
            figures.append(fig20)

            st.write("---")

            # 19. Crime Count by Police District
            crime_counts_district = crime['Police District Name'].value_counts().reset_index()
            crime_counts_district.columns = ['Police District Name', 'Crime Count']

            fig21 = plt.figure(figsize=(12, 6))
            scatter = plt.scatter(x=crime_counts_district['Police District Name'],
                                  y=crime_counts_district['Crime Count'],
                                  c=crime_counts_district['Crime Count'], cmap='coolwarm', s=100, edgecolors='black')

            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Number of Crimes")
            plt.title("Crime Counts by Police District")
            plt.colorbar(scatter, label='Number of Crimes')
            plt.tight_layout()
            st.pyplot(fig21)
            figures.append(fig21)

            st.write("---")

            # 20. Heat map with longitude and latitude
            st.subheader("Heat map with longitude and latitude")
            m_1 = folium.Map(location=[39.1377, -77.13593], tiles="openstreetmap", zoom_start=10)
            HeatMap(data=crime[["Latitude", "Longitude"]].dropna(), radius=10).add_to(m_1)
            st_folium(m_1, width=700)
            st.write("\n")
            st.write("\n")
            st.write("\n")

            col1, col2, col3 = st.columns([7.5, 5, 4])

            with col2:
                if st.button("Create PDF"):
                    pdf_path = generate_pdf(figures)
                    st.success("PDF created with success!")

                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF", f, file_name="crime_report.pdf", mime="application/pdf")

        elif analysis_type == "Analysis of Police Response Time":
            st.subheader("Analysis of Police Response Time")

            # Average Police Response Time per Police Department
            average_response_time = crime.groupby('Police District Name')['Response Time'].mean().reset_index()
            color_resp = sns.color_palette("deep")[5]
            fig22 = plt.figure(figsize=(12, 6))
            plt.fill_between(average_response_time['Police District Name'], average_response_time['Response Time'], color=color_resp, alpha=0.5)
            plt.plot(average_response_time['Police District Name'], average_response_time['Response Time'],
                     color=color_resp, linewidth=2)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Average Response Time in Seconds")
            plt.title("Average Police Response Time per Department")
            plt.tight_layout()
            st.pyplot(fig22)
            figures.append(fig22)

            st.write("---")

            # Average Police Response Time by Crime Type
            crime_response_time_by_type = crime.groupby('Crime Name1')['Response Time'].mean().sort_values(
                ascending=False)
            fig23 = plt.figure(figsize=(12, 6))
            crime_response_time_by_type.plot(kind='barh', color=color_resp)
            plt.title('Average Police Response Time by Crime Type', fontsize=16)
            plt.xlabel('Average Response Time (seconds)', fontsize=14)
            plt.ylabel('Crime Type', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig23)
            figures.append(fig23)
            st.write("\n")
            st.write("\n")
            st.write("\n")

            col1, col2, col3 = st.columns([7.5, 5, 4])

            with col2:
                if st.button("Create PDF"):
                    pdf_path = generate_pdf(figures)
                    st.success("PDF created with success!")

                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF", f, file_name="crime_report.pdf", mime="application/pdf")

        elif analysis_type == "Analysis of Crime Victims":
            st.subheader("Analysis of Crime Victims")

            # Victims Distribution per Crime
            fig24 = plt.figure(figsize=(12, 8))
            top_crimes_victims = crime.groupby('Crime Name1')['Victims'].sum().sort_values(ascending=False).head(10)
            pal_seaborn_deep_victims = sns.color_palette("deep")[1]
            top_crimes_victims.plot(kind='area', stacked=True, color=pal_seaborn_deep_victims, alpha=0.7,
                                    title='Victims Distribution per Crime')
            plt.ylabel("Number of victims")
            st.pyplot(fig24)
            figures.append(fig24)

            st.write("---")

            # Average Number of Victims per Crime
            avg_victims_per_crime = crime.groupby("Crime Name1")["Victims"].mean().sort_values(ascending=False)
            categories = avg_victims_per_crime.index
            values = avg_victims_per_crime.values

            N = len(categories)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]
            values_closed = np.append(values, values[0])
            pal_seaborn_deep = sns.color_palette("deep")[0]

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
            st.pyplot(fig)
            figures.append(fig)
            st.write("\n")
            st.write("\n")
            st.write("\n")

            col1, col2, col3 = st.columns([7.5, 5, 4])

            with col2:
                if st.button("Create PDF"):
                    pdf_path = generate_pdf(figures)
                    st.success("PDF created with success!")

                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF", f, file_name="crime_report.pdf", mime="application/pdf")

    elif selected_analysis == "Prediction":
        st.header("5. Crime Type Prediction")
        st.write("cute description using gradient boosting blablabla")

        df_pred = crime[["Start_Date_Time",
                         "End_Date_Time",
                         "Dispatch Date / Time",
                         "Crime Name1",
                         "Police District Name"]].dropna()
        df_pred["response_time"] = (df_pred["Dispatch Date / Time"] - df_pred["Start_Date_Time"]).dt.total_seconds()
        df_pred["crime_duration"] = (df_pred["End_Date_Time"] - df_pred["Start_Date_Time"]).dt.total_seconds()
        df_pred["Hour"] = df_pred["Start_Date_Time"].dt.hour
        df_pred["Year"] = df_pred["Start_Date_Time"].dt.year

        top_crimes = df_pred["Crime Name1"].value_counts().nlargest(10).index
        df_pred = df_pred[df_pred["Crime Name1"].isin(top_crimes)].sample(n=10000, random_state=42)

        le_crime = LabelEncoder()
        le_district = LabelEncoder()
        df_pred["Crime_Label"] = le_crime.fit_transform(df_pred["Crime Name1"])
        df_pred["District_Code"] = le_district.fit_transform(df_pred["Police District Name"])

        X = df_pred[["response_time", "crime_duration", "Hour", "Year", "District_Code"]]
        y = df_pred["Crime_Label"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gb_clf.fit(X_scaled, y)

        if "prediction_result" not in st.session_state:
            st.session_state.prediction_result = None
        if "show_evaluation" not in st.session_state:
            st.session_state.show_evaluation = False

        with st.form("prediction_form"):
            st.subheader("Occurrence details:")

            response_time_min = st.number_input("Police response time (minutes)", min_value=0.0)
            crime_duration_min = st.number_input("Crime duration (minutes)", min_value=0.0)
            hour_input = st.number_input("Hour of the day (0-23)", min_value=0, max_value=23)
            year_input = st.number_input("Year", min_value=2016, max_value=2020)
            district_input = st.selectbox("Police district", options=list(le_district.classes_))

            submitted = st.form_submit_button("Predict Type of Crime")

        if submitted:
            try:
                response_time_input = response_time_min * 60
                crime_duration_input = crime_duration_min * 60
                district_code = le_district.transform([district_input])[0]

                user_data = pd.DataFrame([{
                    "response_time": response_time_input,
                    "crime_duration": crime_duration_input,
                    "Hour": hour_input,
                    "Year": year_input,
                    "District_Code": district_code
                }])

                user_scaled = scaler.transform(user_data)
                prediction = gb_clf.predict(user_scaled)[0]
                prediction_proba = gb_clf.predict_proba(user_scaled)[0]

                crime_name = le_crime.inverse_transform([prediction])[0]
                confidence = round(100 * prediction_proba[prediction], 2)

                st.session_state.prediction_result = (crime_name, confidence)
                st.session_state.show_evaluation = False  # Reset avaliação

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.prediction_result = None

        if st.session_state.prediction_result:
            crime_name, confidence = st.session_state.prediction_result
            st.success(f"Type of crime predicted: **{crime_name}**")
            st.info(f"Accuracy: **{confidence}%**")

            if st.button("Show model evaluation"):
                st.session_state.show_evaluation = True

        if st.session_state.show_evaluation:
            st.subheader("Confusion Matrix")
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            gb_clf.fit(X_train, y_train)
            y_pred = gb_clf.predict(X_test)

            fig_conf = plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix - Gradient Boosting")
            st.pyplot(fig_conf)

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=le_crime.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)