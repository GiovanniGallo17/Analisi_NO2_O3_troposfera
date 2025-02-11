# Analisi_NO2_O3_troposfera
# Analisi Concentrazione NO2 e O3

## Descrizione del Progetto
Questo script Python esegue un'analisi dettagliata delle concentrazioni di biossido di azoto (NO2) e ozono (O3) utilizzando file NetCDF. L'analisi include la visualizzazione dei dati, calcolo delle statistiche e analisi delle correlazioni.

## Requisiti
- Python 3.7+
- Librerie richieste:
  - `netCDF4`
  - `numpy`
  - `matplotlib`
  - `scipy`

## Installazione delle Dipendenze
```bash
pip install netCDF4 numpy matplotlib scipy
```

## File Richiesti
- `no2_Gen23_100m.nc`: Dataset NetCDF per NO2
- `o3_Gen23_100m.nc`: Dataset NetCDF per O3

## Funzionalità Principali
1. Stampa informazioni dettagliate sui dataset NetCDF
2. Visualizzazione delle concentrazioni giornaliere
3. Calcolo e analisi delle correlazioni tra NO2 e O3

## Come Eseguire lo Script
```bash
python analisi_concentrazione.py
```

## Funzioni Principali
- `print_netcdf_info()`: Mostra dettagli e statistiche dei dataset
- `plot_daily_concentrations()`: Genera grafici delle concentrazioni
- `calculate_correlations()`: Calcola la correlazione tra serie di dati
- `analyze_correlation()`: Analisi dettagliata della correlazione
- `analyze_detailed_correlation()`: Analisi avanzata con statistiche aggiuntive

## Nota
Lo script è configurato per analizzare dati relativi alla zona di Londra (coordinate approssimative: Lat 51.506848, Lon -0.125368)

## Licenza
[Specificare la licenza del progetto]

## Autore
[Nome dell'Autore]
