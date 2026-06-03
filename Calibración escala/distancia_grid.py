import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pathlib import Path

#Ajustar la prominence y la distance hasta que sean los mínimos que queremos

#%%
# Ruta al csv
ruta_csv = "/Users/Mauri/Downloads/perfiles grilla /perfil 6.csv"

# Leer archivo
df = pd.read_csv(ruta_csv)

# Primera columna: píxeles
x = df.iloc[:, 0].to_numpy()

# Segunda columna: señal
y = df.iloc[:, 1].to_numpy()

# Buscar mínimos relativos
# prominence ayuda a ignorar ruido
indices_minimos, _ = find_peaks(-y, prominence=6000, distance = 50)

# Posiciones de los mínimos en píxeles
minimos_x = x[indices_minimos]

# Diferencias entre mínimos consecutivos
diferencias = np.diff(minimos_x)

# Promedio del período
periodo_promedio = np.mean(diferencias)
desv_std = np.std(diferencias, ddof=1)
error_media = desv_std / np.sqrt(len(diferencias))


#print("Posiciones de los mínimos:")
#print(minimos_x)

#print("\nDiferencias entre mínimos consecutivos:")
#print(diferencias)

print(f"Período promedio = ({periodo_promedio:.2f} ± {error_media:.2f}) px")
#print(f"Desviación estándar = {desv_std:.2f} px")
#print(f"Número de períodos = {len(diferencias)}")


#%%


plt.figure(figsize=(8,4))
plt.plot(x, y, label='Señal', color = 'grey')
plt.plot(minimos_x, y[indices_minimos], 'ko', label='Mínimos')
plt.xlabel('Píxeles')
plt.ylabel('Señal')
plt.legend()
plt.show()


#%% Hacerlo para varios perfiles

# Carpeta con los csv
carpeta = Path("/Users/Mauri/Downloads/perfiles grilla ")

# Lista donde guardar todas las diferencias
todas_las_diferencias = []

# Opcional: guardar resultados archivo por archivo
resultados = {}

for archivo in carpeta.glob("*.csv"):

    # Leer csv
    df = pd.read_csv(archivo)

    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()

    # Buscar mínimos
    indices_minimos, _ = find_peaks(
        -y,
        prominence=6000, distance = 50
    )

    minimos_x = x[indices_minimos]

    # Diferencias entre mínimos
    diferencias = np.diff(minimos_x)

    # Guardar para este archivo
    resultados[archivo.name] = diferencias

    # Agregar a la lista global
    todas_las_diferencias.extend(diferencias)

# Convertir a numpy array
todas_las_diferencias = np.array(todas_las_diferencias)

# Estadísticas globales
promedio_global = np.mean(todas_las_diferencias)
desv_std_global = np.std(todas_las_diferencias, ddof=1)
error_media_global = desv_std_global / np.sqrt(len(todas_las_diferencias))

print(f"Promedio global = {promedio_global:.3f} px")
#print(f"Desv. estándar = {desv_std_global:.3f} px")
#print(f"Error de la media = {error_media_global:.3f} px")
print(f"Número total de períodos = {len(todas_las_diferencias)}")

print(f"Período promedio = ({promedio_global:.2f} ± {error_media_global:.2f}) px")

#%%para ver qué dió cada archivo

for nombre, diferencias in resultados.items():

    promedio = np.mean(diferencias)
    error = np.std(diferencias, ddof=1) / np.sqrt(len(diferencias))

    print(f"{nombre}: {promedio:.2f} ± {error:.2f} px")
    
    