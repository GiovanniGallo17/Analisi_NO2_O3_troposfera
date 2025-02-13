import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Percorsi dei file
no2_file = 'no2_Gen23_100m.nc'
o3_file = 'o3_Gen23_100m.nc'

# Coordinate approssimative di Londra 
LONDRA_LAT = 51.506848
LONDRA_LON = -0.125368

# Funzione per leggere file NC
def leggi_netcdf(filename):
    dataset = nc.Dataset(filename)
    return dataset

# Trova l'indice del valore più vicino in un array (per trovare gli indici relativi a due valori di latitudine e longitudine)
def cerca_indice(array, valore):
    array = np.asarray(array)
    idx = (np.abs(array - valore)).argmin()
    return idx

# Visualizza le concentrazioni orarie per un giorno specifico
def plot_concentrazioni(no2_data, o3_data, day=0):
    # Estrai i dati per il giorno specificato (24 ore)
    start_idx = day * 24
    end_idx = start_idx + 24
    
    no2_daily = no2_data[start_idx:end_idx]
    o3_daily = o3_data[start_idx:end_idx]
    
    # Crea array delle ore
    hours = np.arange(24)
    
    # Creazione del grafico
    plt.figure(figsize=(15, 8))
    
    # Plot delle concentrazioni orarie
    plt.plot(hours, no2_daily, 'ro-', label='NO2', linewidth=2, markersize=8)
    plt.plot(hours, o3_daily, 'bo-', label='O3', linewidth=2, markersize=8)
    
    plt.title(f'Concentrazioni orarie degli inquinanti a Londra - Giorno {day+1} Gennaio 2023')
    plt.xlabel('Ora del giorno')
    plt.ylabel('Concentrazione (µg/m³)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Aggiungi etichette per ogni ora
    plt.xticks(hours)
    
    # Aggiungi annotazioni per i valori massimi e minimi
    no2_max_idx = np.argmax(no2_daily)
    no2_min_idx = np.argmin(no2_daily)
    o3_max_idx = np.argmax(o3_daily)
    o3_min_idx = np.argmin(o3_daily)
    
    # Annotazioni NO2
    plt.annotate(f'Max NO2: {no2_daily[no2_max_idx]:.1f}', 
                xy=(no2_max_idx, no2_daily[no2_max_idx]), 
                xytext=(10, 10), textcoords='offset points')
    plt.annotate(f'Min NO2: {no2_daily[no2_min_idx]:.1f}', 
                xy=(no2_min_idx, no2_daily[no2_min_idx]), 
                xytext=(10, -15), textcoords='offset points')
    
    # Annotazioni O3
    plt.annotate(f'Max O3: {o3_daily[o3_max_idx]:.1f}', 
                xy=(o3_max_idx, o3_daily[o3_max_idx]), 
                xytext=(-10, 10), textcoords='offset points')
    plt.annotate(f'Min O3: {o3_daily[o3_min_idx]:.1f}', 
                xy=(o3_min_idx, o3_daily[o3_min_idx]), 
                xytext=(-10, -15), textcoords='offset points')
    
    plt.tight_layout()
    return plt.gcf()

# Calcola e stampa la correlazione tra due serie di dati
def calcola_correlazione(data1, data2, name1, name2):
    correlazione, p_value = stats.pearsonr(data1, data2)
    print(f"\nCorrelazione tra {name1} e {name2}:")
    print(f"Coefficiente di correlazione: {correlazione:.3f}")
    print(f"P-value: {p_value:.2e}")
    return correlazione, p_value
   
def main():
    # Lettura dei dataset
    no2_dataset = leggi_netcdf(no2_file)
    o3_dataset = leggi_netcdf(o3_file)

    # Estrazione delle coordinate
    lats = no2_dataset.variables['lat'][:]
    lons = no2_dataset.variables['lon'][:]
    
    # Trova gli indici più vicini per Londra
    lat_idx = cerca_indice(lats, LONDRA_LAT)
    lon_idx = cerca_indice(lons, LONDRA_LON)
    
    print(f"\nCoordinate selezionate per Londra:")
    print(f"Latitudine: {lats[lat_idx]} (indice: {lat_idx})")
    print(f"Longitudine: {lons[lon_idx]} (indice: {lon_idx})")
    
    # Estrazione delle serie temporali per Londra
    no2_data = no2_dataset.variables['no2'][:, lat_idx, lon_idx]
    o3_data = o3_dataset.variables['o3'][:, lat_idx, lon_idx]
    
    print("\nStatistiche generali dei dati estratti per Londra:")
    print("=" * 50)
    print("\nNO2:")
    print(f"Range completo: [{np.min(no2_data):.2f}, {np.max(no2_data):.2f}] µg/m³")
    print(f"Media totale: {np.mean(no2_data):.2f} µg/m³")
    
    print("\nO3:")
    print(f"Range completo: [{np.min(o3_data):.2f}, {np.max(o3_data):.2f}] µg/m³")
    print(f"Media totale: {np.mean(o3_data):.2f} µg/m³")
    
    # Plot delle concentrazioni per i primi 3 giorni (scegliere i giorni da visualizzare)
    for day in range(3):
        fig = plot_concentrazioni(no2_data, o3_data, day)
        plt.show()
        
        # Stampa statistiche giornaliere
        start_idx = day * 24
        end_idx = start_idx + 24
        no2_daily = no2_data[start_idx:end_idx]
        o3_daily = o3_data[start_idx:end_idx]
        print(f"\nStatistiche per il Giorno {day+1} Gennaio 2023 - Londra")
        print("=" * 50)
        
        print("\nNO2:")
        print(f"Media giornaliera: {np.mean(no2_daily):.2f} µg/m³")
        print(f"Massimo: {np.max(no2_daily):.2f} µg/m³ (ora {np.argmax(no2_daily)})")
        print(f"Minimo: {np.min(no2_daily):.2f} µg/m³ (ora {np.argmin(no2_daily)})")
        
        print("\nO3:")
        print(f"Media giornaliera: {np.mean(o3_daily):.2f} µg/m³")
        print(f"Massimo: {np.max(o3_daily):.2f} µg/m³ (ora {np.argmax(o3_daily)})")
        print(f"Minimo: {np.min(o3_daily):.2f} µg/m³ (ora {np.argmin(o3_daily)})")
        
        # Analisi della correlazione
        print("\nAnalisi della correlazione giornaliera:")
        print("=" * 50)
        calcola_correlazione(no2_daily, o3_daily, "NO2", "O3")
        
    # Analisi della correlazione per l'intero mese/periodo
    print("\nAnalisi della correlazione per l'intero mese:")
    print("=" * 50)

    correlazione, p_value = stats.pearsonr(no2_data, o3_data)
    print(f"Correlazione: {correlazione:.3f} (p-value: {p_value:.2e})")
    
    # Analisi per fasce orarie
    ore = np.arange(len(no2_data)) % 24
    day_periods = {
        'Notte (00-06)': (0, 6),
        'Mattina (06-12)': (6, 12),
        'Pomeriggio (12-18)': (12, 18),
        'Sera (18-24)': (18, 24)
    }
    
    print("\nCorrelazione per fasce orarie:")
    for period, (start, end) in day_periods.items():
        mask = (ore >= start) & (ore < end)
        period_corr, period_p = stats.pearsonr(no2_data[mask], o3_data[mask])
        print(f"{period}: {period_corr:.3f} (p-value: {period_p:.2e})")

if __name__ == "__main__":
    main()
