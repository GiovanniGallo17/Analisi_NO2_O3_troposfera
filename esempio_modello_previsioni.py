import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from analisi_concentrazioni import leggi_netcdf, cerca_indice

# Percorsi dei file
no2_file = 'no2_Gen23_100m.nc'
o3_file = 'o3_Gen23_100m.nc'

# Coordinate approssimative di Londra (circa)
LONDRA_LAT = 51.506848
LONDRA_LON = -0.125368

def normalizza_dati(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def prepara_dati(no2_data, o3_data):
    # Combina i dati di NO2 e O3 in un unico array
    X = np.column_stack((no2_data, o3_data))
    
    # Normalizza i dati 
    X_normalized, X_mean, X_std = normalizza_dati(X)
    
    # Crea i target 
    y = X_normalized.copy()
    
    return X_normalized, y, X_mean, X_std

# Definizione della rete neurale
class AirQualityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AirQualityPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def plot_previsioni(no2_reale, o3_reale, no2_previsto, o3_previsto, title="Confronto tra valori reali e previsti"):
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(no2_reale, label="NO2 Reale", color='blue', alpha=0.7)
    plt.plot(no2_previsto, label="NO2 Previsto", color='red', linestyle='--', alpha=0.7)
    plt.title(f"{title} - NO2")
    plt.xlabel("Tempo")
    plt.ylabel("Concentrazione (µg/m³)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    plt.plot(o3_reale, label="O3 Reale", color='green', alpha=0.7)
    plt.plot(o3_previsto, label="O3 Previsto", color='orange', linestyle='--', alpha=0.7)
    plt.title(f"{title} - O3")
    plt.xlabel("Tempo")
    plt.ylabel("Concentrazione (µg/m³)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

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
    
    # Estrazione delle serie temporali per Londra
    no2_data = no2_dataset.variables['no2'][:, lat_idx, lon_idx]
    o3_data = o3_dataset.variables['o3'][:, lat_idx, lon_idx]
    
    # Prepara i dati
    X, y, X_mean, X_std = prepara_dati(no2_data, o3_data)
    
    # Converti i dati in tensori PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Parametri della rete neurale
    input_size = 2  # NO2 e O3
    hidden_size = 64 # Layer rete
    output_size = 2  # Predici NO2 e O3
    modello = AirQualityPredictor(input_size, hidden_size, output_size)
    
    # Addestramento della rete neurale
    criterio = nn.MSELoss()
    ottimizzatore = optim.Adam(modello.parameters(), lr=0.001)
    
    epoche = 100
    for epoca in range(epoche):
        outputs = modello(X_tensor)
        loss = criterio(outputs, y_tensor)
        ottimizzatore.zero_grad()
        loss.backward()
        ottimizzatore.step()
        
        if (epoca + 1) % 10 == 0:
            print(f'Epoca [{epoca+1}/{epoche}], Loss: {loss.item():.4f}')
    
    # Salva il modello addestrato
    torch.save(modello.state_dict(), 'modello_previsioni.pth')
    
    # Previsioni 
    modello.eval()
    with torch.no_grad():
        previsioni = modello(X_tensor).numpy()
    
    # Denormalizzazione delle previsioni
    previsioni_denorm = previsioni * X_std + X_mean
    reali_denorm = y * X_std + X_mean
    
    # Estrai NO2 e O3 dalle previsioni
    no2_previsto = previsioni_denorm[:, 0]
    o3_previsto = previsioni_denorm[:, 1]
    no2_reale = reali_denorm[:, 0]
    o3_reale = reali_denorm[:, 1]
    
    # Plotta le previsioni 
    plot_previsioni(no2_reale, o3_reale, no2_previsto, o3_previsto, title="Previsioni 2023")

if __name__ == "__main__":
    main()
