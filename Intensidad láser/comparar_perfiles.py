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


#%%
def analizar_perfil(ruta_tiff,
                    half_size, mm,
                    x0, y0, centro_imagen,
                    ajuste_gauss,centro_ajuste,
                    band_halfwidth,
                    graficar_perfiles = True
                    ):
        
        
    img = iio.imread(ruta_tiff).astype(float)
    pixel_size = 5.86e-3#mm
    
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
    
    if ajuste_gauss:
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
        
    
    
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    
    plt.rcParams.update({
        "font.size": 14,        # tamaño base
        "axes.labelsize": 16,   # labels x e y
        "axes.titlesize": 18,   # títulos
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14
    })
    
    if graficar_perfiles:

        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[0.25, 4, 1],
            height_ratios=[1, 4]
        )
    
        ax_top   = fig.add_subplot(gs[0, 1])
        ax_cbar  = fig.add_subplot(gs[1, 0])
        ax_img   = fig.add_subplot(gs[1, 1])
        ax_right = fig.add_subplot(gs[1, 2])
        
    else:
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[0.25, 4]
        )
    
        ax_cbar = fig.add_subplot(gs[0, 0])
        ax_img  = fig.add_subplot(gs[0, 1])
    
        ax_top = None
        ax_right = None
        
    
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
    xc = x0
    yc = y0
    # centro del ajuste en coordenadas GLOBALES
    if centro_ajuste:
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


    if graficar_perfiles:
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
    
    
    if mm:
        # Convertir automáticamente px -> mm en los ejes
        ax_img.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x * pixel_size:.1f}")
        )
        
        ax_img.yaxis.set_major_formatter(
            FuncFormatter(lambda y, pos: f"{y * pixel_size:.1f}")
        )
        
        ax_img.set_xlabel("x [mm]")
        ax_img.set_ylabel("y [mm]")
        
    if centro_imagen:
        ax_img.plot(x0,y0,marker="+",color="cyan",markersize=15,mew=3)
        
    
    ax_img.axhline(ymin, color="grey", ls="-")
    ax_img.axhline(ymax, color="grey", ls="-")
    
    ax_img.axvline(xmin, color="grey", ls="-")
    ax_img.axvline(xmax, color="grey", ls="-")
    
    
    plt.show()



    #Realizar 


    #Pruebo a hacer el return como un diccionario, nunca lo había hecho, pero es más cómodo por el tema de los ifs
    return {
        "popt": popt if ajuste_gauss else None,
        "pcov": pcov if ajuste_gauss else None,
    
        "xc": xc,
        "yc": yc,
    
        "perfil_x": {
            "x": np.arange(x1, x2),
            "I": perfil_x
        },
    
        "perfil_y": {
            "y": np.arange(y1, y2),
            "I": perfil_y
        }
    }



#%%
analizar_perfil(ruta_tiff = "/Users/Mauri/Downloads/top_hat_0.tiff",
                    half_size = 700, mm = True,
                    x0 = 1000, y0 = 600, centro_imagen = False,
                    band_halfwidth = 5,
                    ajuste_gauss = True,
                    centro_ajuste = True,
                    graficar_perfiles = True
                    )


# half size: mitad del lado del cuadrado a recortar, en px
#x0, y0, coordenadas para centrar el recorte allí
#centro_imagen = True grafica el centro de la imagen con un +
#band_halfwidth, ancho de las regiones tomadas para calcular los perfiles
#ajuste_gauss = True realiza un ajuste gaussiano del perfil
#centro_ajuste = True los perfiles tienen como centro el centro de la gaussiana ajustada
# graficar_perfiles = True grafica los perfiles


#Para acceder al return

# resultado = analizar_perfil(...)
# popt = resultado["popt"] # 
# x_perfil = resultado["perfil_x"]["x"]
# I_x = resultado["perfil_x"]["I"]
# y_perfil = resultado["perfil_y"]["y"]
# I_y = resultado["perfil_y"]["I"]
# xc = resultado["xc"] # centro del perfil en x
# yc = resultado["yc"] # centro del perfil en y

#El ajuste toma los datos del recorte solamente, es decir si se cambia el recorte, puede no estar ajustando bien
#Capaz cambiar algunos parámetros para que no sea tan confuso, por ej que si centro_ajuste = True y ajuste = false, tira error

#%%Comparo Perfiles


top_hat = analizar_perfil(ruta_tiff = "/Users/Mauri/Downloads/top_hat_0.tiff",
                    half_size = 700, mm = True,
                    x0 = 1000, y0 = 600, centro_imagen = False,
                    band_halfwidth = 5,
                    ajuste_gauss = True,
                    centro_ajuste = True,
                    graficar_perfiles = True
                    )

x_perfil_hat = top_hat["perfil_x"]["x"]
I_x_hat = top_hat["perfil_x"]["I"]
# y_perfil = resultado["perfil_y"]["y"]
# I_y = resultado["perfil_y"]["I"]
xc_hat = top_hat["xc"]

#%%
gauss = analizar_perfil(ruta_tiff = "/Users/Mauri/Downloads/sin_nada_1.tiff",
                    half_size = 700, mm = True,
                    x0 = 1000, y0 = 600, centro_imagen = False,
                    band_halfwidth = 5,
                    ajuste_gauss = True,
                    centro_ajuste = True,
                    graficar_perfiles = True
                    )

x_perfil_gauss = gauss["perfil_x"]["x"]
I_x_gauss = gauss["perfil_x"]["I"]
xc_gauss = gauss["xc"]

#%%
def gauss_1d(x, A, x0, sigma, C):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + C
C0 = np.min(I_x_gauss)
A0 = np.max(I_x_gauss) - C0
x0_0 = x_perfil_gauss[np.argmax(I_x_gauss)]
sigma0 = len(x_perfil_gauss) / 6
p0 = (A0, x0_0, sigma0, C0)
popt, pcov = curve_fit(gauss_1d, x_perfil_gauss,I_x_gauss, p0=p0) 
A_fit, x0_fit, sigma_fit, C_fit = popt
x_fit = np.linspace(x_perfil_gauss.min(), x_perfil_gauss.max(), 1000)
I_fit = gauss_1d(x_fit, *popt)


#%%

pixel_size = 5.86e-3  # mm por pixel (si corresponde)

plt.figure(figsize=(6,4))

plt.plot((x_perfil_hat - xc_hat)* pixel_size, I_x_hat, color="darkviolet", label = "Perfil luego del PiShaper")
plt.plot((x_perfil_gauss - xc_gauss)* pixel_size, I_x_gauss, color="darkcyan", alpha = 0.5, label = "Perfil antes del PiShaper")
plt.plot((x_fit - xc_gauss)* pixel_size, I_fit, "b-", label="Ajuste del perfil antes del PiShaper")


plt.xlabel("x [mm]")
plt.ylabel("Intensidad [a.u.]")
plt.legend(loc='upper right', bbox_to_anchor=(2, 1))
plt.title("Perfil en X")
plt.grid(True, alpha = 0.3)

plt.show()


#%%

analizar_perfil(ruta_tiff = "/Users/Mauri/Downloads/sin_nada_1.tiff",
                    half_size = 700, mm = True,
                    x0 = 1000, y0 = 600, centro_imagen = False,
                    band_halfwidth = 5,
                    ajuste_gauss = False,
                    centro_ajuste = False,
                    graficar_perfiles = False
                    )






