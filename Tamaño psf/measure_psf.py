import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


#%%

#-----------------------------
# Función gaussiana
#-----------------------------
def gauss(x, A, x0, sigma, C):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + C


#-----------------------------
# Carpeta con los CSV
#-----------------------------
carpeta = "/Users/Mauri/Downloads/psf_15_perfiles"

archivos = sorted(glob.glob(os.path.join(carpeta, "*.csv")))

fwhm_lista = []
nombres = []

factor = 2 * np.sqrt(2 * np.log(2))  # FWHM = factor*sigma


for archivo in archivos:

    # Leer CSV
    datos = pd.read_csv(archivo)

    # Primera columna: x
    # Segunda columna: intensidad
    x = datos.iloc[:, 0].to_numpy()
    y = datos.iloc[:, 1].to_numpy()

    # Valores iniciales
    A0 = y.max() - y.min()
    x00 = x[np.argmax(y)]
    sigma0 = (x.max() - x.min()) / 10
    C0 = y.min()

    p0 = [A0, x00, sigma0, C0]

    try:
        popt, pcov = curve_fit(gauss, x, y, p0=p0)

        sigma = abs(popt[2])
        fwhm = factor * sigma

        fwhm_lista.append(fwhm)
        nombres.append(os.path.basename(archivo))

        print(f"{os.path.basename(archivo)}  FWHM = {fwhm:.3f} px")

    except RuntimeError:
        print(f"No se pudo ajustar {archivo}")


#-----------------------------
# Estadística
#-----------------------------
fwhm_lista = np.array(fwhm_lista)

N = len(fwhm_lista)

promedio = np.mean(fwhm_lista)

desvio = np.std(fwhm_lista, ddof=1)

error_promedio = desvio / np.sqrt(N)


psf = promedio * 6.5 / 50.95
err_psf = error_promedio* 6.5 / 59.64

print("\n==============================")
print(f"N = {N}")
print(f"FWHM promedio = {promedio:.4f} px")
print(f"Error estadístico = {error_promedio:.4f} px")
print(  f"ancho PSF = ({psf:.3f} ± {error_promedio:.3f})µm")
print("==============================")



