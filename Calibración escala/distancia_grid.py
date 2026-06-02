import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

#Ajustar la prominence y la distance hasta que sean los mínimos que queremos

#%%
# Ruta al csv
ruta_csv = "/Users/Mauri/Downloads/perfil_grilla.csv"

# Leer archivo
df = pd.read_csv(ruta_csv)

# Primera columna: píxeles
x = df.iloc[:, 0].to_numpy()

# Segunda columna: señal
y = df.iloc[:, 1].to_numpy()

# Buscar mínimos relativos
# prominence ayuda a ignorar ruido
indices_minimos, _ = find_peaks(-y, prominence=4000)

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


