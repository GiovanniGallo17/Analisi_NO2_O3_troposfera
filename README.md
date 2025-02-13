
## Descrizione del Progetto
Questo repository contiene tre script Python per l'analisi e la modellizzazione dei dati di qualità dell'aria, concentrandosi su biossido di azoto (NO2) e ozono (O3):

### Script Principali
1. **analisi_concentrazione.py**
   - Analisi dettagliata delle concentrazioni di NO2 e O3
   - Visualizzazione dei dati
   - Calcolo della correlazione

2. **analisi_verticale.py**
   - Analisi verticale dei dati atmosferici
   - Calcolo del lapse rate
   - Correlazione tra O3, NO2 e temperatura
   - Utilizzo di xarray e pandas per l'elaborazione dei dati

3. **modello2.py**
   - Modello di Machine Learning per la previsione della qualità dell'aria
   - Utilizzo di PyTorch per la costruzione di una rete neurale
   - Normalizzazione e preparazione dei dati
   - Predizione dei livelli di NO2 e O3

## Requisiti
- Python 3.7+
- Librerie richieste elencate in `requirements.txt`

## Installazione delle Dipendenze
Utilizzare il file `requirements.txt` per installare tutte le dipendenze necessarie:
```bash
pip install -r requirements.txt
```

## File di Input Richiesti
- `no2_Gen23_100m.nc`: Dataset NetCDF per NO2
- `o3_Gen23_100m.nc`: Dataset NetCDF per O3
- `data_plev.nc`: Dataset per l'analisi verticale

## Come Eseguire gli Script
```bash
# Analisi concentrazioni
python analisi_concentrazione.py

# Analisi verticale
python analisi_verticale.py

# Modello predittivo
python modello2.py
```

## Funzioni Principali
### analisi_concentrazione.py
- `print_netcdf_info()`: Mostra dettagli e statistiche dei dataset
- `plot_daily_concentrations()`: Genera grafici delle concentrazioni
- `calculate_correlations()`: Calcola la correlazione tra serie di dati
- `analyze_correlation()`: Analisi dettagliata della correlazione
- `analyze_detailed_correlation()`: Analisi avanzata con statistiche aggiuntive

### analisi_verticale.py
- Calcolo del lapse rate atmosferico
- Analisi delle correlazioni verticali tra gas e temperatura

### modello2.py
- `normalize_data()`: Normalizzazione dei dati di input
- `AirQualityPredictor`: Rete neurale per la previsione della qualità dell'aria

## Nota
Gli script sono configurati per analizzare dati relativi alla zona di Londra (coordinate approssimative: Lat 51.506848, Lon -0.125368)

