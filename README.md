
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
   - Correlazione tra O3, NO2 e lapse rate

3. **modello_previsioni.py**
   - Modello di Machine Learning per la previsione delle concentrazioni dei due inquinanti
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
Dataset per NO2 e O3 (CAMS European air quality reanalyses), scaricabile da https://ads.atmosphere.copernicus.eu/datasets/cams-europe-air-quality-reanalyses?tab=overview 

Dataset per l'analisi verticale (CAMS global atmospheric composition forecasts), scaricabile da https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts?tab=overview

## Come Eseguire gli Script
```bash
python analisi_concentrazione.py

python analisi_verticale.py

python modello_previsioni.py
```
## Nota
Il primo script è onfigurato per analizzare dati relativi alla zona di Londra (coordinate approssimative: Lat 51.506848, Lon -0.125368) 

