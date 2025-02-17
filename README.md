
## Descrizione 
Questa repository contiene tre script Python per l'analisi e la modellizzazione dei dati di qualità dell'aria, concentrandosi su biossido di azoto (NO2) e ozono (O3):

### Script Principali
1. **analisi_concentrazioni.py**
   - Analisi dettagliata delle concentrazioni giornaliere di NO2 e O3
   - Calcolo della correlazione tra i due inquinanti
   - Visualizzazione dei dati

2. **analisi_verticale.py**
   - Analisi verticale dei dati atmosferici
   - Calcolo dell'adiabatic lapse rate
   - Correlazione tra O3, NO2 e lapse rate
   - Visualizzazione dei dati

3. **esempio_modello_previsioni.py**
   - Esempio di modello per la previsione delle concentrazioni dei due inquinanti
   - Utilizzo di PyTorch per la costruzione di una rete neurale 
   - Normalizzazione e preparazione dei dati
   - Predizione dei livelli di NO2 e O3 con l'utilizzo di dati simulati

## Requisiti
- Python 3.7+
- Librerie richieste:
  - numpy
  - pandas
  - scipy
  - netCDF4
  - xarray
  - torch
  - scikit-learn
  - matplotlib

## Installazione delle Dipendenze
Per installare tutte le dipendenze necessarie:
```bash
pip install numpy pandas scipy netCDF4 xarray torch scikit-learn matplotlib
```

## File di Input Richiesti
Dataset per NO2 e O3 (CAMS European air quality reanalyses), scaricabile da https://ads.atmosphere.copernicus.eu/datasets/cams-europe-air-quality-reanalyses?tab=overview 

Dataset per l'analisi verticale (CAMS global atmospheric composition forecasts), scaricabile da https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts?tab=overview

## Come Eseguire gli Script
```bash
python analisi_concentrazioni.py

python analisi_verticale.py

python esempio_modello_previsioni.py
```
## Nota
Il primo script è configurato per analizzare dati relativi alla zona di Londra (coordinate approssimative: Lat 51.506848, Lon -0.125368).

Il secondo script è  per analizzare dati relativi al Nord Europa, in particolare la zona tra Londra-Parigi-Amsterdam
