#Láser Coherent OBIS 640nm XT 300mW
#Cámara IDS uEye CP Rev. 2.2


## ----- idea:

# sacar 1000 fotos en formato tiff
# hacer un loop que ajuste las mil por una gaussiana, y que guarde el valor del sigma con el error

#sacar foto







#análisis
import imageio.v3 as iio
import numpy as np

img = iio.imread("foto_0001.tif")
img = img.astype(float)  # importante para ajuste
print(img.shape, img.dtype)



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




#crear array xy y array de intensidades

from scipy.optimize import curve_fit

ny, nx = img.shape

x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)

xdata = (X.ravel(), Y.ravel())
ydata = img.ravel()




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

A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, C_fit = popt

print("A =", A_fit)
print("x0 =", x0_fit)
print("y0 =", y0_fit)
print("sigma_x =", sigma_x_fit)
print("sigma_y =", sigma_y_fit)
print("theta =", theta_fit)
print("C =", C_fit)



#FWHM
fwhm_x = 2.355 * sigma_x_fit
fwhm_y = 2.355 * sigma_y_fit

print("FWHM_x =", fwhm_x)
print("FWHM_y =", fwhm_y)







#Graficamos para comparar:

import matplotlib.pyplot as plt

plt.figure(figsize=(7,6))
plt.imshow(roi, origin="lower", cmap="inferno")
plt.colorbar(label="Intensidad")

# contornos del ajuste
levels = np.linspace(np.min(fit_roi), np.max(fit_roi), 8)
plt.contour(fit_roi, levels=levels, colors="cyan", linewidths=1)

plt.scatter([x0_fit], [y0_fit], color="lime", marker="x", s=100, label="Centro ajustado")
plt.title("ROI con contornos del ajuste Gaussiano 2D")
plt.xlabel("x (pix)")
plt.ylabel("y (pix)")
plt.legend()
plt.show()


#desp agregar lo de comparar con la gaussiana ajustada

