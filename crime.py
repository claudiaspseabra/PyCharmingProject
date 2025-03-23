import pandas as pd

#-----------------AZUL-------------------

crime = pd.read_csv("Crime.csv", low_memory=False)
crime = crime[crime["State"] == "MD"]
print(crime.dtypes)

# 1. Mode 
mode_cr_number = crime['CR Number'].mode()[0]
mode_victims = crime['Victims'].mode()[0]
crime['Offence Code'] = pd.to_numeric(crime['Offence Code'], errors='coerce')
mode_offence_code = crime['Offence Code'].mode()[0]

print(f"Mode of Offence Code: {mode_offence_code}")
print("\nMode of CR Number:", mode_cr_number)
print("Mode of Victims:", mode_victims)

# 2. Range 
range_cr_number = crime['CR Number'].max() - crime['CR Number'].min()
range_victims = crime['Victims'].max() - crime['Victims'].min()

print("Range of CR Number:", range_cr_number)
print("Range of Victims:", range_victims)

# 3. Median 
median_cr_number = crime['CR Number'].median()
median_victims = crime['Victims'].median()

print("\nMedian of CR Number:", median_cr_number)
print("Median of Victims:", median_victims)

# 4. Mean 
mean_victims = crime['Victims'].mean()

print("\nMean of Victims:", mean_victims)

# 5. Max
max_cr_number = crime['CR Number'].max()
max_victims = crime['Victims'].max()

print("Max of CR Number:", max_cr_number)
print("Max of Victims:", max_victims)

# 6. Min 
min_cr_number = crime['CR Number'].min()
min_victims = crime['Victims'].min()

print("Min of CR Number:", min_cr_number)
print("Min of Victims:", min_victims)

# 7. Quartiles
quartiles_victims = crime['Victims'].quantile([0.25, 0.5, 0.75])

print("\nQuartiles of Victims:")
print(quartiles_victims)



#--------------------VERMELHO---------------------~

#Crime Name 1
count_crime1 = crime["Crime Name1"].value_counts()
print(count_crime1)

#Mode
print("\n")
print("Mode: ")
mode_crime_name1 = count_crime1.idxmax()
max_crime1_count = count_crime1.max()
print(f"{mode_crime_name1}: {max_crime1_count}")

# Crime Name 3
print("---------------------")
print("\n")
count_crime3 = crime["Crime Name3"].value_counts()
print(count_crime3)

#Mode
print("\n")
print("Mode: ")
mode_crime_name3 = count_crime3.idxmax()
max_crime3_count = count_crime3.max()
print(f"{mode_crime_name3}: {max_crime3_count}")
#print(count_crime3.max())

# Police District Name
print("---------------------")
print("\n")
count_police_district = crime["Police District Name"].value_counts()
print(count_police_district)

#Mode
print("\n")
print("Mode: ")
mode_police_district = count_police_district.idxmax()
max_pdcount = count_police_district.max()
print(f"{mode_police_district}: {max_pdcount}")


#----------------------ROXO-------------------------------

# Agency
print("---------------------")
print("\n")
count_agency = crime["Agency"].value_counts()
print(count_agency)

#Mode
print("\n")
print("Mode: ")
mode_angency = count_agency.idxmax()
max_agencycount = count_agency.max()
print(f"{mode_angency}: {max_agencycount}")

# Place
print("---------------------")
print("\n")
count_place = crime["Place"].value_counts()
print(count_place)

#Mode
print("\n")
print("Mode: ")
mode_place = count_place.idxmax()
max_placecount = count_place.max()
print(f"{mode_place}: {max_placecount}")


# Street Type
print("---------------------")
print("\n")
count_strtype = crime["Street Type"].value_counts()
print(count_strtype)

#Mode
print("\n")
print("Mode: ")
#mode_strtype = count_strtype.mode()
mode_strtype = crime["Street Type"].mode()
max_strtypecount = count_strtype.max()
print(mode_strtype)
print(f"{mode_strtype}: {max_strtypecount}")

#print(crime["Beat"].nunique())
print("All modes:")
print(crime.mode().iloc[0])

