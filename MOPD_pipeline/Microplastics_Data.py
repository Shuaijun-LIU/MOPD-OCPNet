import pandas as pd

df = pd.read_csv('./Data/NOAA_Microplastics/Marine_Microplastics.csv')

df = df[['Latitude', 'Longitude', 'Density Range', 'Density Class', 'Oceans', 'Date']]

df.fillna(0, inplace=True)

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

df_sorted = df.sort_values(by='Date', ascending=False)
df_latest = df_sorted.drop_duplicates(subset=['Latitude', 'Longitude'], keep='first')

df_latest.to_csv('./Data/NOAA_Microplastics/Marine_Microplastics_cleaned.csv', index=False)

print("Microplastics data saved as Marine_Microplastics_cleaned.csv")
