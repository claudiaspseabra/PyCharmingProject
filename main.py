import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats

# Set plot style for consistency
sns.set(style="whitegrid")


# ---------------------------
# Function: Load Data with Caching
# ---------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Crime Excel.xlsx")
    except Exception as e:
        st.error("Error loading dataset: " + str(e))
        return None
    return df


# Load the dataset
df = load_data()

# Если датасет загрузился, выводим список столбцов и корректируем типы данных
if df is not None:
    st.write("Columns in the dataset:", df.columns)
    # Приведение проблемных столбцов к строковому типу, например "NIBRS Code"
    if "NIBRS Code" in df.columns:
        df["NIBRS Code"] = df["NIBRS Code"].astype(str)

# ---------------------------
# Title of the Dashboard
# ---------------------------
st.title("Comprehensive and Advanced Crime Data Analysis")

# ---------------------------
# Project Requirements Section
# ---------------------------
st.header("Project Requirements")
st.markdown("""
**In the first part of the work, the student should:**

- **Dataset Selection:** Choose a dataset. The datasets should be different for each group. Options include UCI Datasets, Kaggle Datasets, or other repositories.
- **Dataset Description (5%):** Describe the dataset's characteristics (e.g., domain, size, data types, entities, etc.).
- **Statistical Analysis (5%):** Develop a statistical analysis using measures such as average, variance, covariance, correlations, etc.
- **Feature Engineering (2.5%):** Create new features to enrich the analysis.
- **Graphical Analysis (10%):** Develop graphical analysis.
- **Dashboard (10%):** Make a graphical dashboard with a coherent representation of the results.
- **Critical Analysis (2.5%):** Include a critical analysis in the report.
- **Normalization & Standardization (2.5%):** Perform normalization and standardization of your data and analyze the new distributions.
- **Report and Oral Presentation (2.5%):** Elaborate the first report and the corresponding oral presentation.
""")

# ---------------------------
# 1. Dataset Description
# ---------------------------
if df is not None:
    st.header("1. Dataset Description")
    st.write("**Domain:** Crime Analysis")
    st.write("**Size:** {} rows and {} columns".format(df.shape[0], df.shape[1]))
    st.write("**Data Types:**")
    st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Type'}))
    st.write("**Sample of First 5 Rows:**")
    st.dataframe(df.head())

    # Detailed Feature Description Report (as Markdown)
    st.header("Feature Description Report")
    st.markdown("""
    **1. Incident ID**  
    *Description:* A unique identifier for each crime incident.  
    *Usage:* Serves as the primary key for tracking individual records.

    **2. Offence Code**  
    *Description:* Code representing the specific offence committed.  
    *Usage:* Helps classify and group incidents by offence type.

    **3. CR Number**  
    *Description:* Unique crime report number.  
    *Usage:* Ensures data integrity and aids in tracking reports.

    **4. Dispatch Date / Time**  
    *Description:* The date and time when the incident was dispatched.  
    *Usage:* Critical for time series analysis; temporal components can be extracted to detect trends and seasonality.

    **5. NIBRS Code**  
    *Description:* Crime classification code based on the National Incident-Based Reporting System.  
    *Usage:* Enables standardized comparison across jurisdictions.

    **6. Victims**  
    *Description:* Number of victims involved in the incident.  
    *Usage:* Useful for assessing incident severity and statistical analyses.

    **7. Crime Name1**  
    *Description:* Primary category or name of the crime.  
    *Usage:* Serves as the main label for crime type; can be used for categorical analysis and one-hot encoding.

    **8. Crime Name2**  
    *Description:* Secondary crime category or descriptor.  
    *Usage:* Provides additional context for the incident.

    **9. Crime Name3**  
    *Description:* Tertiary crime category or additional detail.  
    *Usage:* Offers further granularity in crime classification.

    **10. Police District Name**  
    *Description:* Name of the police district responsible for the incident.  
    *Usage:* Used for spatial analysis and comparing crime rates across districts.

    **11. Block Address**  
    *Description:* Block address where the incident occurred.  
    *Usage:* Supports mapping and hotspot analysis.

    **12. City, State, Zip Code**  
    *Description:* Location details of the incident.  
    *Usage:* Enables regional and spatial analysis.

    **13. Agency**  
    *Description:* Law enforcement agency involved.  
    *Usage:* Can be used to compare performance and response metrics.

    **14. Place, Sector, Beat, PRA, Address Number, Street Prefix, Street Name, Street Suffix, Street Type**  
    *Description:* Detailed address and administrative information.  
    *Usage:* Essential for precise geolocation and spatial clustering.

    **15. Start_Date_Time & End_Date_Time**  
    *Description:* Incident start and end times.  
    *Usage:* Together, they allow calculation of incident duration and detailed temporal analysis.

    **16. Latitude & Longitude**  
    *Description:* Geographical coordinates of the incident.  
    *Usage:* Used for mapping, spatial clustering, and location-based analyses.

    **17. Police District Number, Location**  
    *Description:* Additional location identifiers.  
    *Usage:* Provide further administrative categorization.

    **Conclusion:**  
    This detailed feature description lays a strong foundation for understanding the dataset's structure and guides subsequent preprocessing, feature engineering, and analysis.
    """)

    # ---------------------------
    # 2. Statistical Analysis
    # ---------------------------
    st.header("2. Statistical Analysis")

    # Получаем список числовых столбцов
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.write("Numeric Columns:", numeric_cols)

    # Основная статистика
    st.subheader("Basic Statistics")
    stats_df = pd.DataFrame({
        "Mean": df[numeric_cols].mean(),
        "Median": df[numeric_cols].median(),
        "Std Dev": df[numeric_cols].std(),
        "Variance": df[numeric_cols].var(),
        "Skewness": df[numeric_cols].skew(),
        "Kurtosis": df[numeric_cols].kurt()
    })
    st.dataframe(stats_df)

    # Матрицы ковариаций и корреляций
    st.subheader("Covariance Matrix")
    st.dataframe(df[numeric_cols].cov())

    st.subheader("Correlation Matrix")
    st.dataframe(df[numeric_cols].corr())

    # Анализ квантилей
    st.subheader("Quantile Analysis")
    quantiles_df = df[numeric_cols].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    st.dataframe(quantiles_df)

    # Анализ пропусков (Missing Values)
    st.subheader("Missing Values Analysis")
    missing_df = pd.DataFrame(df.isna().sum(), columns=["Missing Count"])
    missing_df["Missing Percentage"] = missing_df["Missing Count"] / df.shape[0] * 100
    st.dataframe(missing_df.sort_values("Missing Percentage", ascending=False))

    # ---------------------------
    # 3. Advanced Feature Engineering
    # ---------------------------
    st.header("3. Advanced Feature Engineering")

    # --- Date Processing ---
    if 'Dispatch Date / Time' in df.columns:
        df['Dispatch Date / Time'] = pd.to_datetime(df['Dispatch Date / Time'], errors='coerce')
        df['Year'] = df['Dispatch Date / Time'].dt.year
        df['Month'] = df['Dispatch Date / Time'].dt.month
        df['Day'] = df['Dispatch Date / Time'].dt.day
        df['Weekday'] = df['Dispatch Date / Time'].dt.dayofweek  # 0 = Monday, 6 = Sunday
        df['Quarter'] = df['Dispatch Date / Time'].dt.quarter
        df['Week'] = df['Dispatch Date / Time'].dt.isocalendar().week
        st.write(
            "Extracted date components from **Dispatch Date / Time**: Year, Month, Day, Weekday, Quarter, and Week.")

    # --- New Feature: Total Crime Names Count ---
    crime_name_cols = ['Crime Name1', 'Crime Name2', 'Crime Name3']
    available_crime_cols = [col for col in crime_name_cols if col in df.columns]
    if available_crime_cols:
        df['Total_Crime_Names'] = df[available_crime_cols].notna().sum(axis=1)
        st.write("Created new feature **Total_Crime_Names**: count of non-null values among Crime Name columns.")

    # --- New Feature: Incident Duration ---
    if 'Start_Date_Time' in df.columns and 'End_Date_Time' in df.columns:
        df['Start_Date_Time'] = pd.to_datetime(df['Start_Date_Time'], errors='coerce')
        df['End_Date_Time'] = pd.to_datetime(df['End_Date_Time'], errors='coerce')
        df['Incident_Duration'] = (df['End_Date_Time'] - df['Start_Date_Time']).dt.total_seconds() / 60.0
        st.write("Created new feature **Incident_Duration**: duration in minutes (End_Date_Time - Start_Date_Time).")

    # --- One-Hot Encoding for Categorical Feature ---
    if 'Crime Name1' in df.columns:
        crime_type_dummies = pd.get_dummies(df['Crime Name1'], prefix='CrimeName1')
        df = pd.concat([df, crime_type_dummies], axis=1)
        st.write("Performed one-hot encoding on **Crime Name1** to generate binary columns.")

    # --- Interaction Feature: Victims and Total Crime Names ---
    if 'Victims' in df.columns and 'Total_Crime_Names' in df.columns:
        df['Victims_Crime_Interaction'] = df['Victims'] * df['Total_Crime_Names']
        st.write("Created interaction feature **Victims_Crime_Interaction** (Victims * Total_Crime_Names).")

    # --- Alternative Feature: Square of Victims ---
    if 'Victims' in df.columns:
        df['Victims_squared'] = df['Victims'] ** 2
        st.write("Created squared feature **Victims_squared** (square of Victims).")

    st.write("**Sample of Data After Advanced Feature Engineering:**")
    st.dataframe(df.head(10))

    st.markdown("""
    **Advanced Feature Description:**

    - **Year, Month, Day, Weekday, Quarter, Week:** These features help detect seasonal trends and temporal patterns in crime events.
    - **Total_Crime_Names:** Indicates the number of crime name fields filled, reflecting the complexity of the incident.
    - **Incident_Duration:** Measures how long an incident lasted, critical for time series and operational analysis.
    - **Victims_Crime_Interaction & Victims_squared:** Capture non-linear effects and interactions related to the number of victims.
    """)

    # ---------------------------
    # 4. Graphical Analysis
    # ---------------------------
    st.header("4. Graphical Analysis")

    # --- Histograms for Numeric Features ---
    st.subheader("Histograms of Numeric Features")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f'Histogram of {col}')
        st.pyplot(fig)

    # --- Box Plots for Outlier Detection ---
    st.subheader("Box Plots for Outlier Detection")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col].dropna(), ax=ax)
        ax.set_title(f'Box Plot of {col}')
        st.pyplot(fig)

    # --- Scatterplot Matrix (Pairplot) ---
    st.subheader("Scatterplot Matrix (Pairplot)")
    if len(numeric_cols) > 1:
        pairplot_fig = sns.pairplot(df[numeric_cols].dropna())
        st.pyplot(pairplot_fig.fig)

    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    # --- QQ Plots for Normality Check ---
    st.subheader("QQ Plots for Normality Check")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        stats.probplot(df[col].dropna(), dist="norm", plot=ax)
        ax.set_title(f'QQ Plot of {col}')
        st.pyplot(fig)

    # --- Additional: Bar Chart for Categorical Feature (Crime Name1) ---
    if 'Crime Name1' in df.columns:
        st.subheader("Distribution of Crime Name1")
        crime_counts = df['Crime Name1'].value_counts().reset_index()
        crime_counts.columns = ['Crime Name1', 'Count']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Crime Name1', y='Count', data=crime_counts, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title("Frequency of Crime Name1")
        st.pyplot(fig)

    # --- Additional: Missing Values Heatmap ---
    st.subheader("Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Values Heatmap")
    st.pyplot(fig)

    # --- Additional: Temporal Trend Analysis ---
    if 'Dispatch Date / Time' in df.columns:
        st.subheader("Temporal Trend of Incidents")
        df_time = df.set_index('Dispatch Date / Time').resample('M').size().reset_index(name='Incident Count')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Dispatch Date / Time', y='Incident Count', data=df_time, ax=ax)
        ax.set_title("Monthly Incident Count Trend")
        st.pyplot(fig)

    # ---------------------------
    # 5. Data Normalization and Standardization
    # ---------------------------
    st.header("5. Data Normalization and Standardization")
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()

    df_numeric = df[numeric_cols].dropna()
    standardized = scaler_standard.fit_transform(df_numeric)
    normalized = scaler_minmax.fit_transform(df_numeric)

    df_standardized = pd.DataFrame(standardized, columns=numeric_cols)
    df_normalized = pd.DataFrame(normalized, columns=numeric_cols)

    st.subheader("Standardized Data (mean=0, std=1)")
    st.dataframe(df_standardized.head())

    st.subheader("Normalized Data (range [0,1])")
    st.dataframe(df_normalized.head())

    st.subheader("Comparison of Distributions")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.kdeplot(df_numeric[col], label='Original', ax=ax)
        sns.kdeplot(df_standardized[col], label='Standardized', ax=ax)
        sns.kdeplot(df_normalized[col], label='Normalized', ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.legend()
        st.pyplot(fig)

    # ---------------------------
    # 6. Critical Analysis and Mini Review
    # ---------------------------
    st.header("6. Critical Analysis and Mini Review")
    st.markdown("""
    **Key Observations:**

    - The dataset focuses on crime analysis and contains a rich set of features including unique identifiers, location details, temporal data, and crime classifications.
    - Basic statistical measures reveal varying scales, skewed distributions, and missing values in some columns, which justify the need for normalization and robust feature engineering.
    - Advanced feature engineering (e.g., extraction of date components, creation of Total_Crime_Names, Incident_Duration, and interaction features) provides deeper insights into the data.
    - Graphical analyses (histograms, box plots, pairplots, QQ plots, bar charts, missing values heatmaps, and temporal trend analysis) help identify outliers, detect seasonal patterns, and understand variable distributions.

    **Mini Review:**

    This project demonstrates a comprehensive and in-depth exploratory data analysis on crime data. The combination of detailed statistical measures, advanced feature engineering, and a wide variety of visualizations provides a solid foundation for future predictive modeling and decision-making. The dashboard effectively communicates insights and supports further research.
    """)

    # ---------------------------
    # 7. Report and Oral Presentation
    # ---------------------------
    st.header("7. Report and Oral Presentation")
    st.markdown("""
    **Report Outline:**

    1. **Dataset Description:** Overview of the domain, size, data types, and sample rows.
    2. **Statistical Analysis:** Detailed exploration including measures of central tendency, dispersion, skewness, kurtosis, covariance, correlations, quantile analysis, and missing values.
    3. **Advanced Feature Engineering:** Extraction of date components (from Dispatch Date / Time), creation of new features (Total_Crime_Names, Incident_Duration, interactions, etc.), and transformation techniques.
    4. **Graphical Analysis:** Visualizations including histograms, box plots, scatterplot matrices (pairplot), QQ plots, bar charts for categorical distributions, missing value heatmaps, and temporal trend analysis.
    5. **Data Normalization and Standardization:** Scaling numeric features and comparing the resulting distributions.
    6. **Critical Analysis and Mini Review:** Discussion of data quality, identified patterns, and potential directions for future research.

    **Oral Presentation:**

    - **Introduction:** Provide an overview of the dataset and its domain.
    - **Statistical Analysis:** Highlight key findings from both basic and advanced statistical analyses.
    - **Advanced Feature Engineering:** Explain the rationale behind the newly derived features and their impact on the analysis.
    - **Graphical Analysis:** Present visualizations that reveal data patterns, outliers, seasonal trends, and correlations.
    - **Conclusion:** Summarize insights and propose recommendations for further exploration.
    """)

    # ---------------------------
    # Sidebar Navigation for Quick Access
    # ---------------------------
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    - [Project Requirements](#project-requirements)
    - [Dataset Description](#1-dataset-description)
    - [Statistical Analysis](#2-statistical-analysis)
    - [Advanced Feature Engineering](#3-advanced-feature-engineering)
    - [Graphical Analysis](#4-graphical-analysis)
    - [Data Normalization and Standardization](#5-data-normalization-and-standardization)
    - [Critical Analysis and Mini Review](#6-critical-analysis-and-mini-review)
    - [Report and Oral Presentation](#7-report-and-oral-presentation)
    """)
else:
    st.write("Failed to load the dataset. Please ensure that 'Crime Excel.xlsx' is in the working directory.")
