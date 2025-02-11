import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Percorsi dei file
no2_file = 'no2_Gen23_100m.nc'
o3_file = 'o3_Gen23_100m.nc'

# Coordinate approssimative di Londra (circa)
LOCATION_LAT = 51.506848
LOCATION_LON = -0.125368

def print_netcdf_info(dataset, name):
    """
    Stampa informazioni dettagliate sul dataset NetCDF
    """
    print(f"\nInformazioni dettagliate sul dataset {name}:")
    print("=" * 50)
    
    # Stampa le variabili disponibili e i loro attributi
    print("\nVariabili disponibili:")
    for var_name, var in dataset.variables.items():
        print(f"\nVariabile: {var_name}")
        print(f"Dimensioni: {var.dimensions}")
        print(f"Forma: {var.shape}")
        if hasattr(var, 'units'):
            print(f"Unità di misura: {var.units}")
        if hasattr(var, 'long_name'):
            print(f"Nome completo: {var.long_name}")
        
        # Se è la variabile principale, mostra alcune statistiche
        if var_name in ['no2', 'o3']:
            data = var[:]
            print(f"Statistiche:")
            print(f"- Min: {np.min(data)}")
            print(f"- Max: {np.max(data)}")
            print(f"- Media: {np.mean(data)}")
            print(f"- Deviazione standard: {np.std(data)}")

def read_netcdf_data(filename):
    dataset = nc.Dataset(filename)
    return dataset

def find_nearest_index(array, value):
    """Trova l'indice del valore più vicino in un array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_daily_concentrations(no2_data, o3_data, day=0):
    """
    Visualizza le concentrazioni orarie per un giorno specifico
    """
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

def calculate_correlations(data1, data2, name1, name2):
    """
    Calcola e stampa la correlazione tra due serie di dati
    """
    correlation, p_value = stats.pearsonr(data1, data2)
    print(f"\nCorrelazione tra {name1} e {name2}:")
    print(f"Coefficiente di correlazione: {correlation:.3f}")
    print(f"P-value: {p_value:.2e}")
    return correlation, p_value

def analyze_correlation(data1, data2, name1, name2):
    """
    Analisi dettagliata della correlazione tra due serie di dati
    """
    # Calcola correlazione di Pearson
    correlation, p_value = stats.pearsonr(data1, data2)
    
    # Calcola correlazione di Spearman (non parametrica)
    spearman_corr, spearman_p = stats.spearmanr(data1, data2)
    
    # Crea il grafico di dispersione
    plt.figure(figsize=(10, 6))
    plt.scatter(data1, data2, alpha=0.5)
    plt.xlabel(f'{name1} (µg/m³)')
    plt.ylabel(f'{name2} (µg/m³)')
    plt.title(f'Correlazione tra {name1} e {name2}')
    
    # Aggiungi linea di regressione
    z = np.polyfit(data1, data2, 1)
    p = np.poly1d(z)
    plt.plot(data1, p(data1), "r--", alpha=0.8)
    
    # Aggiungi testo con statistiche
    stats_text = f'Correlazione di Pearson: {correlation:.3f} (p={p_value:.2e})\n'
    stats_text += f'Correlazione di Spearman: {spearman_corr:.3f} (p={spearman_p:.2e})'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()

def analyze_detailed_correlation(data1, data2, name1, name2):
    """
    Analisi dettagliata della correlazione con statistiche aggiuntive
    """
    # Calcola correlazioni
    pearson_corr, pearson_p = stats.pearsonr(data1, data2)
    spearman_corr, spearman_p = stats.spearmanr(data1, data2)
    
    # Statistiche descrittive
    print(f"\nAnalisi dettagliata della correlazione tra {name1} e {name2}")
    print("=" * 50)
    print("\nStatistiche di correlazione:")
    print(f"Correlazione di Pearson: {pearson_corr:.3f} (p-value: {pearson_p:.2e})")
    print(f"Correlazione di Spearman: {spearman_corr:.3f} (p-value: {spearman_p:.2e})")
    
    # Interpretazione della correlazione
    print("\nInterpretazione:")
    if abs(pearson_corr) > 0.7:
        strength = "forte"
    elif abs(pearson_corr) > 0.3:
        strength = "moderata"
    else:
        strength = "debole"
    
    direction = "positiva" if pearson_corr > 0 else "negativa"
    print(f"- La correlazione è {strength} e {direction}")
    
    if pearson_p < 0.05:
        print("- La correlazione è statisticamente significativa (p < 0.05)")
    else:
        print("- La correlazione non è statisticamente significativa (p > 0.05)")
    
    # Analisi per fasce orarie
    hours = np.arange(len(data1)) % 24
    day_periods = {
        'Notte (00-06)': (0, 6),
        'Mattina (06-12)': (6, 12),
        'Pomeriggio (12-18)': (12, 18),
        'Sera (18-24)': (18, 24)
    }
    
    print("\nCorrelazione per fasce orarie:")
    for period, (start, end) in day_periods.items():
        mask = (hours >= start) & (hours < end)
        period_corr, period_p = stats.pearsonr(data1[mask], data2[mask])
        print(f"{period}: {period_corr:.3f} (p-value: {period_p:.2e})")
    
    # Crea il grafico di dispersione avanzato
    plt.figure(figsize=(12, 8))
    
    # Plot principale
    plt.scatter(data1, data2, alpha=0.5, c=hours, cmap='viridis')
    plt.colorbar(label='Ora del giorno')
    
    # Linea di regressione
    z = np.polyfit(data1, data2, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(data1), max(data1), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8, label=f'Regressione lineare')
    
    plt.xlabel(f'{name1} (µg/m³)')
    plt.ylabel(f'{name2} (µg/m³)')
    plt.title(f'Correlazione tra {name1} e {name2} con variazione oraria')
    
    # Statistiche nel grafico
    stats_text = f'Correlazione di Pearson: {pearson_corr:.3f} (p={pearson_p:.2e})\n'
    stats_text += f'Correlazione di Spearman: {spearman_corr:.3f} (p={spearman_p:.2e})\n'
    stats_text += f'Equazione: y = {z[0]:.3f}x + {z[1]:.3f}'
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def main():
    # Lettura dei dataset
    no2_dataset = read_netcdf_data(no2_file)
    o3_dataset = read_netcdf_data(o3_file)
    
    # Stampa informazioni dettagliate sui dataset
    #print_netcdf_info(no2_dataset, "NO2")
    #print_netcdf_info(o3_dataset, "O3")
    
    # Estrazione delle coordinate
    lats = no2_dataset.variables['lat'][:]
    lons = no2_dataset.variables['lon'][:]
    
    # Trova gli indici più vicini per Malta
    lat_idx = find_nearest_index(lats, LOCATION_LAT)
    lon_idx = find_nearest_index(lons, LOCATION_LON)
    
    print(f"\nCoordinate selezionate per Londra:")
    print(f"Latitudine: {lats[lat_idx]} (indice: {lat_idx})")
    print(f"Longitudine: {lons[lon_idx]} (indice: {lon_idx})")
    
    # Estrazione delle serie temporali per Malta
    no2_data = no2_dataset.variables['no2'][:, lat_idx, lon_idx]
    o3_data = o3_dataset.variables['o3'][:, lat_idx, lon_idx]
    
    print("\nStatistiche generali dei dati estratti per Londra:")
    print("=" * 50)
    print("\nNO2:")
    print(f"Range completo: [{np.min(no2_data):.2f}, {np.max(no2_data):.2f}] µg/m³")
    print(f"Media totale: {np.mean(no2_data):.2f} µg/m³")
    print(f"Deviazione standard: {np.std(no2_data):.2f} µg/m³")
    
    print("\nO3:")
    print(f"Range completo: [{np.min(o3_data):.2f}, {np.max(o3_data):.2f}] µg/m³")
    print(f"Media totale: {np.mean(o3_data):.2f} µg/m³")
    print(f"Deviazione standard: {np.std(o3_data):.2f} µg/m³")
    
    # Plot delle concentrazioni per i primi 3 giorni
    for day in range(3):
        fig = plot_daily_concentrations(no2_data, o3_data, day)
        plt.show()
        #plt.savefig(f'malta_concentrations_day_{day+1}.png')
        #plt.close(fig)
        
        # Stampa statistiche giornaliere
        start_idx = day * 24
        end_idx = start_idx + 24
        no2_daily = no2_data[start_idx:end_idx]
        o3_daily = o3_data[start_idx:end_idx]
        print(no2_daily)
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
        calculate_correlations(no2_daily, o3_daily, "NO2", "O3")
        
        # Crea e salva il grafico di correlazione
        fig = analyze_correlation(no2_daily, o3_daily, "NO2", "O3")
        plt.savefig(f'correlazione_NO2_O3_giorno_{day+1}.png')
        plt.close(fig)
    
    # Analisi della correlazione per l'intero periodo
    print("\nAnalisi della correlazione per l'intero periodo:")
    print("=" * 50)
    
    # Analisi dettagliata della correlazione totale
    fig = analyze_detailed_correlation(no2_data, o3_data, "NO2", "O3")
    plt.savefig('correlazione_dettagliata_NO2_O3_totale.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Chiusura dei dataset
    no2_dataset.close()
    o3_dataset.close()

if __name__ == "__main__":
    main()
