#Importar mediciones de fotos,
#Centrar en el centro del  (eligiendolo uno)
#Tomar perfil en las direcciones perpendiculares a ese centro y graficar 
#(opcional)Tomar perfil en radios en torno a ese centro, peomediar y asignar errores, y graficar

import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter


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

#%%Ver imagen y asignar centro

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

# Coordenadas del centro que querés probar
x0 = 850
y0 = 630

plt.figure()
plt.pcolormesh(X, Y, img, shading="auto", cmap="inferno")
plt.colorbar(label="Intensidad")
plt.plot(x0, y0, marker="x", color="cyan", markersize=15, mew=3)
# Opcional: mostrar coordenadas al lado
plt.text(x0 + 10, y0 + 10, f"({x0}, {y0})",
        color="cyan", fontsize=12)

plt.xlabel("x [px]")
plt.ylabel("y [px]")
plt.show()



#%%

img = iio.imread("/Users/Mauri/Downloads/aumentocon200mmy30mm.tiff").astype(float)

half_size = 700

pixel_size = 0 # = 1 si querés la distancia en mm
if pixel_size == 1:
    pixel_size = 5.86e-3#mm
    pixel_flag = 1
else:
    pixel_size = 1


x0 = 1000
y0 = 600
band_halfwidth = 5


# límites del recorte
x1 = max(0, x0 - half_size)
x2 = min(img.shape[1], x0 + half_size)

y1 = max(0, y0 - half_size)
y2 = min(img.shape[0], y0 + half_size)

# recorte
roi = img[y1:y2, x1:x2]



ny, nx = roi.shape
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)

xdata = (X.ravel(), Y.ravel())
ydata = roi.ravel()

# semillas iniciales
C0 = np.min(roi)
A0 = np.max(roi) - C0

#y0_init, x0_init = np.unravel_index(np.argmax(roi), roi.shape)
# centro inicial del ajuste dentro de la ROI
x0_init = x0 - x1
y0_init = y0 - y1


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


#perfil_x = roi.mean(axis=0)
#perfil_y = roi.mean(axis=1)


fig = plt.figure(figsize=(10, 7), constrained_layout=True)

plt.rcParams.update({
    "font.size": 14,        # tamaño base
    "axes.labelsize": 16,   # labels x e y
    "axes.titlesize": 18,   # títulos
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

gs = fig.add_gridspec(
    2, 3,
    width_ratios=[0.25, 4, 1],
    height_ratios=[1, 4]
)

ax_top   = fig.add_subplot(gs[0, 1])
ax_cbar  = fig.add_subplot(gs[1, 0])
ax_img   = fig.add_subplot(gs[1, 1])
ax_right = fig.add_subplot(gs[1, 2])

# Imagen
im = ax_img.imshow(
    roi,
    cmap="inferno",
    origin="lower",
    aspect="auto",
    extent=[x1, x2, y1, y2]
)
ax_img.set_xlabel("x [px]")
ax_img.set_ylabel("y [px]")


# centro del haz dentro de la ROI
# centro del ajuste en coordenadas GLOBALES
xc = int(x0_fit + x1)
yc = int(y0_fit + y1)

# -------------------------
# Perfil en X
# promedio en una banda horizontal
# -------------------------
ymin = max(y1, yc - band_halfwidth)
ymax = min(y2, yc + band_halfwidth)

# -------------------------
# Perfil en Y
# promedio en una banda vertical
# -------------------------
xmin = max(x1, xc - band_halfwidth)
xmax = min(x2, xc + band_halfwidth)


# índices locales dentro de la ROI
xmin_roi = xmin - x1
xmax_roi = xmax - x1

ymin_roi = ymin - y1
ymax_roi = ymax - y1

perfil_x = roi[ymin_roi:ymax_roi, :].mean(axis=0)
perfil_y = roi[:, xmin_roi:xmax_roi].mean(axis=1)

# # Perfil X
# ax_top.plot(x, perfil_x, color="grey")
# ax_top.set_xlim(0, nx-1)
# ax_top.set_xticks([])
# ax_top.set_ylabel("I(x)")

# # Perfil Y
# ax_right.plot(perfil_y, y, color="grey")
# ax_right.set_ylim(0, ny-1)
# ax_right.set_yticks([])
# ax_right.set_xlabel("I(y)")
# ax_right.invert_yaxis()  # para que coincida con origin="lower"

# Perfil X
ax_top.plot(np.arange(x1, x2), perfil_x, color="grey")
ax_top.set_xlim(x1, x2)
ax_top.set_xticks([])
ax_top.set_ylabel("I(x)")

# Perfil Y
ax_right.plot(perfil_y, np.arange(y1, y2), color="grey")
ax_right.set_ylim(y1, y2)
ax_right.set_yticks([])
ax_right.set_xlabel("I(y)")


# Colorbar en su propio eje
cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label("Intensidad")
ax_cbar.yaxis.set_label_position("left")
ax_cbar.yaxis.set_ticks_position("left")


if pixel_flag == 1:
    # Convertir automáticamente px -> mm en los ejes
    ax_img.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x * pixel_size:.1f}")
    )
    
    ax_img.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: f"{y * pixel_size:.1f}")
    )
    
    ax_img.set_xlabel("x [mm]")
    ax_img.set_ylabel("y [mm]")
    
    
ax_img.plot(
    x0,
    y0,
    marker="+",
    color="cyan",
    markersize=15,
    mew=3
)
    

ax_img.axhline(ymin, color="cyan", ls="--")
ax_img.axhline(ymax, color="cyan", ls="--")

ax_img.axvline(xmin, color="lime", ls="--")
ax_img.axvline(xmax, color="lime", ls="--")


plt.show()






#%%Top Hat

img = iio.imread("/Users/Mauri/Downloads/top_hat_0.tiff").astype(float)

half_size = 700

pixel_size = 0 # = 1 si querés la distancia en mm
if pixel_size == 1:
    pixel_size = 5.86e-3#mm
    pixel_flag = 1
else:
    pixel_size = 1


x0 = 1000
y0 = 600
band_halfwidth = 5


# límites del recorte
x1 = max(0, x0 - half_size)
x2 = min(img.shape[1], x0 + half_size)

y1 = max(0, y0 - half_size)
y2 = min(img.shape[0], y0 + half_size)

# recorte
roi = img[y1:y2, x1:x2]



ny, nx = roi.shape
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)

xdata = (X.ravel(), Y.ravel())
ydata = roi.ravel()

# semillas iniciales
C0 = np.min(roi)
A0 = np.max(roi) - C0

#y0_init, x0_init = np.unravel_index(np.argmax(roi), roi.shape)
# centro inicial del ajuste dentro de la ROI
x0_init = x0 - x1
y0_init = y0 - y1


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


#perfil_x = roi.mean(axis=0)
#perfil_y = roi.mean(axis=1)

#%% Analizo el top_hat


fig = plt.figure(figsize=(10, 7), constrained_layout=True)

plt.rcParams.update({
    "font.size": 14,        # tamaño base
    "axes.labelsize": 16,   # labels x e y
    "axes.titlesize": 18,   # títulos
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

gs = fig.add_gridspec(
    2, 3,
    width_ratios=[0.25, 4, 1],
    height_ratios=[1, 4]
)

ax_top   = fig.add_subplot(gs[0, 1])
ax_cbar  = fig.add_subplot(gs[1, 0])
ax_img   = fig.add_subplot(gs[1, 1])
ax_right = fig.add_subplot(gs[1, 2])

# Imagen
im = ax_img.imshow(
    roi,
    cmap="inferno",
    origin="lower",
    aspect="auto",
    extent=[x1, x2, y1, y2]
)
ax_img.set_xlabel("x [px]")
ax_img.set_ylabel("y [px]")


# centro del haz dentro de la ROI
# centro del ajuste en coordenadas GLOBALES
xc = int(x0_fit + x1)
yc = int(y0_fit + y1)

# -------------------------
# Perfil en X
# promedio en una banda horizontal
# -------------------------
ymin = max(y1, yc - band_halfwidth)
ymax = min(y2, yc + band_halfwidth)

# -------------------------
# Perfil en Y
# promedio en una banda vertical
# -------------------------
xmin = max(x1, xc - band_halfwidth)
xmax = min(x2, xc + band_halfwidth)


# índices locales dentro de la ROI
xmin_roi = xmin - x1
xmax_roi = xmax - x1

ymin_roi = ymin - y1
ymax_roi = ymax - y1

perfil_x = roi[ymin_roi:ymax_roi, :].mean(axis=0)
perfil_y = roi[:, xmin_roi:xmax_roi].mean(axis=1)

# # Perfil X
# ax_top.plot(x, perfil_x, color="grey")
# ax_top.set_xlim(0, nx-1)
# ax_top.set_xticks([])
# ax_top.set_ylabel("I(x)")

# # Perfil Y
# ax_right.plot(perfil_y, y, color="grey")
# ax_right.set_ylim(0, ny-1)
# ax_right.set_yticks([])
# ax_right.set_xlabel("I(y)")
# ax_right.invert_yaxis()  # para que coincida con origin="lower"

# Perfil X
ax_top.plot(np.arange(x1, x2), perfil_x, color="grey")
ax_top.set_xlim(x1, x2)
ax_top.set_xticks([])
ax_top.set_ylabel("I(x)")

# Perfil Y
ax_right.plot(perfil_y, np.arange(y1, y2), color="grey")
ax_right.set_ylim(y1, y2)
ax_right.set_yticks([])
ax_right.set_xlabel("I(y)")


# Colorbar en su propio eje
cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label("Intensidad")
ax_cbar.yaxis.set_label_position("left")
ax_cbar.yaxis.set_ticks_position("left")


if pixel_flag == 1:
    # Convertir automáticamente px -> mm en los ejes
    ax_img.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x * pixel_size:.1f}")
    )
    
    ax_img.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: f"{y * pixel_size:.1f}")
    )
    
    ax_img.set_xlabel("x [mm]")
    ax_img.set_ylabel("y [mm]")
    
    
#ax_img.plot(x0,y0,marker="+",color="cyan",markersize=15,mew=3)
    

ax_img.axhline(ymin, color="grey", ls="-")
ax_img.axhline(ymax, color="grey", ls="-")

ax_img.axvline(xmin, color="grey", ls="-")
ax_img.axvline(xmax, color="grey", ls="-")


plt.show()




#Crear fución así es más compacto
#parámetro de mm o de px, del centrado (en torno al x0y,y0 o en torno al del ajuste?),
# del bandwith, del ancho total en píxeles del recorte, y de exportar los perfiles
#así después los grafico juntos.


#%%







