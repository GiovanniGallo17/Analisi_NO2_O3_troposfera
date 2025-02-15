import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analisi_concentrazioni import calcola_correlazione

ds = xr.open_dataset("data_plev.nc", decode_timedelta=True)

# Estrazione variabili dal dataset 
o3 = ds["go3"]  # Ozono
no2 = ds["no2"]  # Diossido di azoto
temperatura = ds["t"]  # Temperatura
livelli_pressione = ds["pressure_level"]  # Livelli di pressione

# Range zona nord Europa
lat = ds.latitude.values
lon = ds.longitude.values
lat_min, lat_max = 48, 52.5  # Limiti di latitudine
lon_min, lon_max = -1, 5.5  # Limiti di longitudine

# Filtrare i dati per l'area specificata
lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]

# Filtrare i dati
o3_filtered = o3.isel(latitude=lat_indices, longitude=lon_indices)
no2_filtered = no2.isel(latitude=lat_indices, longitude=lon_indices)
temp_filtered = temp.isel(latitude=lat_indices, longitude=lon_indices)

# Debug: Verifica le dimensioni dei dati filtrati
print(f"Dimensioni O₃ filtrato: {o3_filtered.shape}")
print(f"Dimensioni NO₂ filtrato: {no2_filtered.shape}")
print(f"Dimensioni Temperatura filtrato: {temp_filtered.shape}")

# Calcolo del lapse rate come dT/dz
dT_dz = np.gradient(temp_filtered, axis=2)  # Derivata lungo l'asse della pressione
lapse_rate = -dT_dz

# Flatten dei dati per poterli analizzare con Pandas
# Utilizzare solo i punti di tempo disponibili
num_time_points = o3_filtered.shape[1]  # 62 punti di tempo

df = pd.DataFrame({
    "o3": o3_filtered.values.flatten()[:num_time_points],
    "no2": no2_filtered.values.flatten()[:num_time_points],
    "lapse_rate": lapse_rate.flatten()[:num_time_points]
})

# Creare una colonna di timestamp
df['timestamp'] = pd.to_datetime(ds.forecast_reference_time.values.flatten()[:num_time_points], unit='s')

# Debug: Verifica le dimensioni del DataFrame
print(f"Dimensioni DataFrame: {df.shape}")

# Ordinare i dati in base al timestamp
df.sort_values('timestamp', inplace=True)

# Calcolo della correlazione tra O3, NO2 e lapse rate
corr, pValue =calcola_correlazione(df["o3"], df["lapse_rate"], "O3", "Lapse Rate")
corr2, pValue2 =calcola_correlazione(df["no2"], df["lapse_rate"], "NO2", "Lapse Rate")

# Creare grafici a linee per le serie temporali
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(df['timestamp'], df['o3'], label='O₃', color='blue')
plt.title('Concentrazione di O₃ nel Tempo')
plt.xlabel('Data')
plt.ylabel('Concentrazione di O₃ (kg/kg)')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(df['timestamp'], df['no2'], label='NO₂', color='orange')
plt.title('Concentrazione di NO₂ nel Tempo')
plt.xlabel('Data')
plt.ylabel('Concentrazione di NO₂ (kg/kg)')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(df['timestamp'], df['lapse_rate'], label='Lapse Rate', color='green')
plt.title('Lapse Rate nel Tempo')
plt.xlabel('Data')
plt.ylabel('Lapse Rate (°C/km)')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()