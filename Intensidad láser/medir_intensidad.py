#Láser Coherent OBIS 640nm XT 300mW
#Cámara IDS uEye CP Rev. 2.2


## ----- idea:

# sacar 1000 fotos en formato tiff
# hacer un loop que ajuste las mil por una gaussiana, y que guarde el valor del sigma con el error

#sacar foto

import sys
print(sys.executable)
# después en bash: "ruta que printeó lo anterior" -m pip install imageio

#%%
# defino la gaussiana
def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, theta, C):
    x, y = coords
    
    x0 = float(x0)
    y0 = float(y0)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    a = (cos_t**2)/(2*sigma_x**2) + (sin_t**2)/(2*sigma_y**2)
    b = -(sin_t*cos_t)/(2*sigma_x**2) + (sin_t*cos_t)/(2*sigma_y**2)
    c = (sin_t**2)/(2*sigma_x**2) + (cos_t**2)/(2*sigma_y**2)

    return A * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2)) + C

def crop_roi_centered_on_max(img, half_size=100):
    """
    Recorta una ROI cuadrada centrada en el pixel de máxima intensidad.
    
    half_size = 100 -> ROI de tamaño (201 x 201)
    """
    ny, nx = img.shape

    y_max, x_max = np.unravel_index(np.argmax(img), img.shape)

    x1 = max(0, x_max - half_size)
    x2 = min(nx, x_max + half_size + 1)

    y1 = max(0, y_max - half_size)
    y2 = min(ny, y_max + half_size + 1)

    roi = img[y1:y2, x1:x2]

    return roi, (x1, x2, y1, y2), (x_max, y_max)

#análisis
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

#%%

img = iio.imread("/Users/Mauri/Downloads/Tiempo de exposición T = 20ms/image_4.tiff")
img = img.astype(float)  # importante para ajuste
print(img.shape, img.dtype)

#crear array xy y array de intensidades
ny, nx = img.shape

x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)

xdata = (X.ravel(), Y.ravel())
ydata = img.ravel()


plt.figure()
plt.pcolormesh(X, Y, img, shading="auto", cmap="inferno")
plt.colorbar(label="Intensidad")
plt.xlabel("x [px]")
plt.ylabel("y [px]")
plt.show()


#%%

#le damos valores iniciales razonables
C0 = np.min(img)
A0 = np.max(img) - C0

y0_init, x0_init = np.unravel_index(np.argmax(img), img.shape)

sigma_x0 = nx / 10
sigma_y0 = ny / 10
theta0 = 0


#Ajustamos
p0 = (A0, x0_init, y0_init, sigma_x0, sigma_y0, theta0, C0)

popt, pcov = curve_fit(gaussian_2d, xdata, ydata, p0=p0)
incertidumbre = np.sqrt(np.diag(pcov))

A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, C_fit = popt

err_sig_x = incertidumbre[3]
err_sig_y = incertidumbre[4]

#%%
print(f"A = {A_fit:.2f}")
print(f"x0 ={x0_fit:.2f}")
print(f"y0 = {y0_fit:.2f}", )
print(f"sigma_x = {sigma_x_fit:.2f}")
print(f"sigma_y = {sigma_y_fit:.2f}")
print(f"theta = {theta_fit:.2f}")
print(f"C = {C_fit:.2f}")

#FWHM
fwhm_x = 2.355 * sigma_x_fit
fwhm_y = 2.355 * sigma_y_fit
print(f"FWHM_x = {fwhm_x:.2f}")
print(f"FWHM_y = {fwhm_y:.2f}")

#%%
#En mm
pixel_size_um = 5.86
pixel_size_mm = pixel_size_um * 1e-3

sigma_x_mm = sigma_x_fit * pixel_size_mm
sigma_y_mm = sigma_y_fit * pixel_size_mm

err_sig_x_mm = err_sig_x * pixel_size_mm
err_sig_y_mm = err_sig_y * pixel_size_mm


fwhm_x_mm = 2.355 * sigma_x_mm
fwhm_y_mm = 2.355 * sigma_y_mm

d_1e2_x = 4 * sigma_x_mm
d_1e2_y = 4 * sigma_y_mm


print(f"sigma_x_mm = ({sigma_x_mm:.2f} ± {err_sig_x_mm:.2f})", "mm")
print(f"sigma_y_mm = ({sigma_y_mm:.2f} ± {err_sig_x_mm:.2f})", "mm")


print(f"diámetro_1/e2_x = {d_1e2_x:.2f}", "mm")
print(f"diámetro_1/e2_y = {d_1e2_y:.2f}", "mm")


print(f"FWHM_x = {fwhm_x_mm:.2f}", "mm")
print(f"FWHM_y = {fwhm_y_mm:.2f}", "mm")


#%% Estadística para el error


sigmas_x = []
sigmas_y = []

carpeta = Path("/Users/Mauri/Downloads/Tiempo de exposición T = 20ms")

# Buscar todos los archivos .tiff (o .tif)
archivos = sorted(list(carpeta.glob("*.tiff"))) + sorted(list(carpeta.glob("*.tif")))

for k, archivo in enumerate(archivos):
    print("Leyendo:", archivo)
    img = iio.imread(archivo)   # <-- esto ya funciona con Path

    img = img.astype(float)  # importante para ajuste
    
    #crear array xy y array de intensidades
    ny, nx = img.shape
    
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    xdata = (X.ravel(), Y.ravel())
    ydata = img.ravel()
    
    if k == 0: # esto lo hago sólo en la 1er iteración, total las fotos son casi iguales
        #le damos valores iniciales razonables
        C0 = np.min(img)
        A0 = np.max(img) - C0
    
        y0_init, x0_init = np.unravel_index(np.argmax(img), img.shape)
    
        sigma_x0 = nx / 10
        sigma_y0 = ny / 10
        theta0 = 0
    
    #Ajustamos
    p0 = (A0, x0_init, y0_init, sigma_x0, sigma_y0, theta0, C0)
    
    popt, pcov = curve_fit(gaussian_2d, xdata, ydata, p0=p0)
    incertidumbre = np.sqrt(np.diag(pcov))
    
    A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, C_fit = popt

    sigmas_x.append(sigma_x_fit)
    sigmas_y.append(sigma_y_fit)
    
    
#%%



#%%



#%% ROI


#%%


img = iio.imread("/Users/Mauri/Downloads/Tiempo de exposición T = 20ms/image_4.tiff").astype(float)

# recortar ROI centrada en el máximo
roi, (x1, x2, y1, y2), (xmax, ymax) = crop_roi_centered_on_max(img, half_size=200)

ny, nx = roi.shape
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)

xdata = (X.ravel(), Y.ravel())
ydata = roi.ravel()

# semillas iniciales
C0 = np.min(roi)
A0 = np.max(roi) - C0

y0_init, x0_init = np.unravel_index(np.argmax(roi), roi.shape)

sigma_x0 = nx / 6
sigma_y0 = ny / 6
theta0 = 0

p0 = (A0, x0_init, y0_init, sigma_x0, sigma_y0, theta0, C0)

# bounds para que no se vaya a cualquier lado (muy recomendable)
lower_bounds = (0, 0, 0, 1e-3, 1e-3, -np.pi/2, -np.inf)
upper_bounds = (np.inf, nx, ny, nx, ny, np.pi/2, np.inf)

popt, pcov = curve_fit(
    gaussian_2d,
    xdata,
    ydata,
    p0=p0,
    bounds=(lower_bounds, upper_bounds)
)

A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, C_fit = popt



plt.figure()
plt.pcolormesh(X, Y, roi, shading="auto", cmap="inferno")
plt.colorbar(label="Intensidad")
plt.xlabel("x [px]")
plt.ylabel("y [px]")
plt.show()



print("==== Ajuste ROI ====")
print(f"A = {A_fit:.2f}")
print(f"x0 ={x0_fit:.2f}")
print(f"y0 = {y0_fit:.2f}", )
print(f"sigma_x = {sigma_x_fit:.2f}")
print(f"sigma_y = {sigma_y_fit:.2f}")
print(f"theta = {theta_fit:.2f}")
print(f"C = {C_fit:.2f}")


# posición del centro en coordenadas de la imagen original
x0_global = x1 + x0_fit
y0_global = y1 + y0_fit

print("Centro global (x,y) =", x0_global, y0_global)





#%%

roi_size = 400
M = 100 # cantidad de mediciones a usar

sigmas_x = []
sigmas_y = []

carpeta = Path("/Users/Mauri/Downloads/Tiempo de exposición T = 20ms")

# Buscar todos los archivos .tiff (o .tif)
archivos = sorted(list(carpeta.glob("*.tiff")) + list(carpeta.glob("*.tif")))[:M]
for k, archivo in enumerate(archivos):
    print("Leyendo:", archivo)
    img = iio.imread(archivo)   # <-- esto ya funciona con Path

    img = img.astype(float)  # importante para ajuste
        
    # recortar ROI centrada en el máximo
    roi, (x1, x2, y1, y2), (xmax, ymax) = crop_roi_centered_on_max(img, half_size = 200)
    
    ny, nx = roi.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    xdata = (X.ravel(), Y.ravel())
    ydata = roi.ravel()
        
    if k == 0: # esto lo hago sólo en la 1er iteración, total las fotos son casi iguales
        #le damos valores iniciales razonables
        C0 = np.min(roi)
        A0 = np.max(roi) - C0
        
        y0_init, x0_init = np.unravel_index(np.argmax(roi), roi.shape)
        
        sigma_x0 = nx / 6
        sigma_y0 = ny / 6
        theta0 = 0
        
        p0 = (A0, x0_init, y0_init, sigma_x0, sigma_y0, theta0, C0)
        
    #Ajustamos
    
    # bounds para que no se vaya a cualquier lado (muy recomendable)
    lower_bounds = (0, 0, 0, 1e-3, 1e-3, -np.pi/2, -np.inf)
    upper_bounds = (np.inf, nx, ny, nx, ny, np.pi/2, np.inf)

    popt, pcov = curve_fit(gaussian_2d,xdata,ydata,p0=p0,
        bounds=(lower_bounds, upper_bounds))

    A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, C_fit = popt
    #incertidumbre = np.sqrt(np.diag(pcov))
    
    #Usamos estos valores como p0 de los siguientes ajustes
    p0 = (A0, x0_init, y0_init, sigma_x0, sigma_y0, theta0, C0)
    
    sigmas_x.append(sigma_x_fit)
    sigmas_y.append(sigma_y_fit)





#%%


sigmas_x1 = sigmas_x
sigmas_y1 = sigmas_y

def media_y_error(datos):
    media = np.mean(datos)
    sigma = np.std(datos, ddof=1)
    error = sigma / np.sqrt(len(datos))
    return media, error

sigma_x, err_sigma_x = media_y_error(sigmas_x1)
sigma_y, err_sigma_y = media_y_error(sigmas_y1)

print(f"Sigma_x: {sigma_x:.4f} ± {err_sigma_x:.4f}")
print(f"Sigma_y: {sigma_y:.4f} ± {err_sigma_y:.4f}")


# En distancias:
pixel_size_um = 5.86
pixel_size_mm = pixel_size_um * 1e-3
sigma_x_mm = sigma_x * pixel_size_mm
sigma_y_mm = sigma_y * pixel_size_mm
err_sigma_x_mm = err_sigma_x * pixel_size_mm
err_sigma_y_mm = err_sigma_y * pixel_size_mm

print(f"sigma_x_mm = ({sigma_x_mm:.6f} ± {err_sig_x_mm:.6f})", "mm")
print(f"sigma_y_mm = ({sigma_y_mm:.6f} ± {err_sig_y_mm:.6f})", "mm")


d_1e2_x = 4 * sigma_x_mm
d_1e2_y = 4 * sigma_y_mm
err_dx = 4 * err_sigma_x_mm
err_dy = 4 * err_sigma_y_mm


print(f"diámetro_1/e2_x = ({d_1e2_x:.4f} ± {err_dx:.4f})", "mm")
print(f"diámetro_1/e2_y = ({d_1e2_y:.4f} ± {err_dy:.4f})", "mm")


#%%

#Graficamos para comparar:


#desp agregar lo de comparar con la gaussiana ajust ada

