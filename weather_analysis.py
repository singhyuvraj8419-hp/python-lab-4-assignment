import pandas as pd   # pandas import

# 1. CSV load karna
df = pd.read_csv("weather.csv")

# 2. Top 5 rows dekhna
print("HEAD:")
print(df.head())

# 3. Dataset information dekhna
print("\nINFO:")
print(df.info())

# 4. Basic statistics
print("\nDESCRIBE:")
print(df.describe())

# ------------ Data Cleaning & Processing ------------

# 1. Missing values hatao (currently hamare data me NaN nahi hai,
#    but assignment ke liye step dikhana zaroori hai)
df = df.dropna()

# 2. Date column ko datetime format me convert karo
df['Date'] = pd.to_datetime(df['Date'])

# 3. Sirf required columns rakho (agar CSV me aur extra columns hon)
df = df[['Date', 'Temperature', 'Humidity', 'Rainfall']]

print("\nCleaned Data:")
print(df.head())
print("\nCleaned Data Info:")
print(df.info())

# ------------ Statistical Analysis with NumPy ------------

import numpy as np

print("\n----- Overall Statistical Analysis -----")
print("Mean Temperature:", np.mean(df['Temperature']))
print("Min Temperature:", np.min(df['Temperature']))
print("Max Temperature:", np.max(df['Temperature']))
print("Std Dev Temperature:", np.std(df['Temperature']))

print("\nMean Humidity:", np.mean(df['Humidity']))
print("Min Humidity:", np.min(df['Humidity']))
print("Max Humidity:", np.max(df['Humidity']))

print("\nTotal Rainfall:", np.sum(df['Rainfall']))
print("Average Rainfall:", np.mean(df['Rainfall']))

# --- Monthly statistics ---
df['Month'] = df['Date'].dt.month

monthly_stats = df.groupby('Month').agg({
    'Temperature': ['mean', 'min', 'max', 'std'],
    'Humidity': ['mean'],
    'Rainfall': ['sum']
})

print("\n----- Monthly Statistics -----")
print(monthly_stats)

# ------------ Visualization with Matplotlib ------------
import matplotlib.pyplot as plt

# Line chart for daily temperature
plt.figure()
plt.plot(df['Date'], df['Temperature'])
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Daily Temperature Trend")
plt.tight_layout()
plt.savefig("daily_temperature.png")
plt.show()

# Bar chart for monthly rainfall
monthly_rain = df.groupby('Month')['Rainfall'].sum()

plt.figure()
monthly_rain.plot(kind='bar')
plt.xlabel("Month")
plt.ylabel("Total Rainfall")
plt.title("Monthly Rainfall")
plt.tight_layout()
plt.savefig("monthly_rainfall.png")
plt.show()

# Scatter plot of humidity vs temperature
plt.figure()
plt.scatter(df['Temperature'], df['Humidity'])
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Humidity vs Temperature")
plt.tight_layout()
plt.savefig("humidity_vs_temperature.png")
plt.show()

# Combined figure
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(df['Date'], df['Temperature'])
plt.title("Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature")

plt.subplot(1, 2, 2)
plt.scatter(df['Temperature'], df['Humidity'])
plt.title("Temp vs Humidity")
plt.xlabel("Temperature")
plt.ylabel("Humidity")

plt.tight_layout()
plt.savefig("combined_plots.png")
plt.show()

# ------------ Grouping & Aggregation ------------

print("\n----- Grouping & Aggregation by Month -----")

grouped = df.groupby('Month').agg({
    'Temperature': 'mean',
    'Humidity': 'mean',
    'Rainfall': 'sum'
})

print(grouped)

# ------------ Export Cleaned Data ------------
df.to_csv("cleaned_weather_data.csv", index=False)
print("\nCleaned data saved to cleaned_weather_data.csv")

