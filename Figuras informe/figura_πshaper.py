
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import affine_transform
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FixedLocator


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

def rotar_roi(roi, angulo_deg, cx, cy):
    theta = np.deg2rad(angulo_deg)

    c = np.cos(theta)
    s = np.sin(theta)

    # Matriz de rotación
    R = np.array([[c, -s],
                  [s,  c]])

    # affine_transform usa la transformación inversa
    M = R.T

    centro = np.array([cy, cx])   # (fila, columna)

    offset = centro - M @ centro

    roi_rot = affine_transform(
        roi,
        matrix=M,
        offset=offset,
        output_shape=roi.shape,
        order=1,
        mode="constant",
        cval=0
    )

    return roi_rot

def analizar_perfil_modificado(ruta_tiff,
                    half_size, mm,
                    x0, y0, centro_imagen,
                    ajuste_gauss,centro_ajuste,
                    band_halfwidth,
                    graficar_perfiles = True,
                    estadística = False,
                    angulo_max = 180,
                    angulo_paso = 5,
                    grafico_primeras_rot = False,
                    grafico_perfiles_x = False
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
    
    
    #Rotación
    # # Centro de la ROI para la rotación
    cx = x0 - x1
    cy = y0 - y1
    

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
    cbar.set_label("Intensidad [a.u.]")
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



    #------Rotaciones------
    
    
    if estadística:
    
        perfiles_x = []
        rois_rotadas = []
        
        angulos = np.arange(0, angulo_max, angulo_paso)
        
        # ======================================
        # Analizar todos los ángulos
        # ======================================
        for angulo in angulos:
        
            roi_rot = rotar_roi(roi, angulo, cx, cy)
            rois_rotadas.append(roi_rot)
        
            perfil_x_rot = roi_rot[ymin_roi:ymax_roi, :].mean(axis=0)
            perfiles_x.append(perfil_x_rot)
        
        perfiles_x = np.array(perfiles_x)
        
        # ======================================
        # Graficar sólo las primeras 12 ROIs
        # ======================================
        if grafico_primeras_rot:
        
            fig, axs = plt.subplots(3, 4, figsize=(12, 9))
            axs = axs.ravel()
            
            for ax, roi_rot, angulo in zip(axs, rois_rotadas[:12], angulos[:12]):
            
                ax.imshow(
                    roi_rot,
                    cmap="inferno",
                    origin="lower",
                    aspect="auto"
                )
            
                ax.set_title(f"{angulo}°")
                ax.axis("off")
            
            plt.tight_layout()
            plt.show()
            
        # ======================================
        # Graficar todos los perfiles
        # ======================================
        if grafico_perfiles_x:

            plt.figure(figsize=(8,6))
            
            cmap = plt.get_cmap("viridis")
            
            for i, angulo in enumerate(angulos):
                plt.plot(
                    np.arange(x1, x2),
                    perfiles_x[i],
                    color=cmap(i/(len(angulos)-1)),
                    label=f"{angulo}°"
                )
            
            plt.xlabel("x [px]")
            plt.ylabel("Intensidad")
            plt.grid(True)
            plt.show()
        

        

    #Pruebo a hacer el return como un diccionario, nunca lo había hecho, pero es más cómodo por el tema de los ifs
    resultado = {
        # Imagen
        "roi": roi,
        "extent": [x1, x2, y1, y2],
    
        # Ajuste
        "popt": popt if ajuste_gauss else None,
        "pcov": pcov if ajuste_gauss else None,
    
        # Centros
        "x0": x0,
        "y0": y0,
        "xc": xc,
        "yc": yc,
    
        # Límites de las bandas
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
    
        # Perfiles
        "perfil_x": {
            "x": np.arange(x1, x2),
            "I": perfil_x
        },
    
        "perfil_y": {
            "y": np.arange(y1, y2),
            "I": perfil_y
        },
    
        # Conversión
        "pixel_size": pixel_size
    }

    if estadística:
        resultado["perfiles_x"] = perfiles_x

    return resultado



#%%Foto sin π-Shaper

gauss = analizar_perfil_modificado(ruta_tiff = "/Users/Mauri/Downloads/antes pishaper_1.tiff",
                    half_size = 900, mm = False,
                    x0 = 921, y0 = 599, centro_imagen = True,
                    band_halfwidth = 5,
                    ajuste_gauss = False,
                    centro_ajuste = False,
                    graficar_perfiles = True,
                    estadística = False,
                    angulo_max = 360,
                    angulo_paso = 10,
                    grafico_perfiles_x = False
                    )

# x_perfil_gauss = gauss["perfil_x"]["x"]
# I_x_gauss = gauss["perfil_x"]["I"]
# xc_gauss = gauss["xc"]
# perfiles_x = gauss["perfiles_x"]

# promedio_gauss = perfiles_x.mean(axis=0)
# error_gauss = perfiles_x.std(axis=0, ddof=1)
# #error_gauss = perfiles_x.std(axis=0, ddof=1) / np.sqrt(perfiles_x.shape[0])

# plt.figure(figsize=(8,6))
# plt.errorbar(x_perfil_gauss[::10], promedio_gauss[::10], yerr=error_gauss[::10], fmt="-", capsize=2)
# plt.xlabel("x [px]")
# plt.ylabel("Intensidad")
# plt.grid(True)
# plt.show()




#%%
fig = plt.figure(figsize=(10, 7), constrained_layout=True)

plt.rcParams.update({
    "font.size": 18,        # tamaño base
    "axes.titlesize": 22,   # títulos
    "axes.labelsize": 24,   # etiquetas de los ejes
    "xtick.labelsize": 18,  # números del eje x
    "ytick.labelsize": 18,  # números del eje y
    "legend.fontsize": 18,  # leyenda (si hay)
})


#graficar_perfiles:

gs = fig.add_gridspec(
    2, 2,
    width_ratios=[0.25, 4],
    height_ratios=[1, 4],
    wspace=0.1
)

ax_top  = fig.add_subplot(gs[0, 1])
ax_cbar = fig.add_subplot(gs[1, 0])
ax_img  = fig.add_subplot(gs[1, 1])


# Imagen
im = ax_img.imshow(
    gauss["roi"],
    cmap="inferno",
    origin="lower",
    aspect="auto",
    extent=gauss["extent"]
)
ax_img.set_xlabel("x [px]")
ax_img.set_ylabel("y [px]")


# centro del haz dentro de la ROI
xc = gauss["xc"]
yc = gauss["yc"]

# centro del ajuste en coordenadas GLOBALES
# if centro_ajuste:
#     xc = int(x0_fit + x1)
#     yc = int(y0_fit + y1)


x1 = gauss["extent"][0]
x2 = gauss["extent"][1]
y1 = gauss["extent"][2]
y2 = gauss["extent"][3]


#if graficar_perfiles:
# Perfil X
ax_top.plot(gauss["perfil_x"]["x"], gauss["perfil_x"]["I"], color="grey")
ax_top.set_xlim(x1, x2)
#ax_top.set_xticks([])
ax_top.tick_params(axis="x", which="both",
                   bottom=False, labelbottom=False)
ax_top.grid(True,which="major",linestyle="--",linewidth=0.8,alpha=0.5)
ax_top.set_ylabel("I(x) [u.a.]", fontsize = 18)


# Colorbar en su propio eje
cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label("Intensidad [u.a.]")
ax_cbar.yaxis.set_label_position("left")
ax_cbar.yaxis.set_ticks_position("left")



# Convertir automáticamente px -> mm en los ejes
pixel_size = gauss["pixel_size"]




ax_img.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{(x - xc) * pixel_size:.0f}")
)

ax_img.yaxis.set_major_formatter(
    FuncFormatter(lambda y, pos: f"{(y - yc) * pixel_size:.0f}")
)


# ticks cada 1 mm
paso_px = 1 / pixel_size
xticks = np.arange(
    xc + np.floor((x1 - xc)/paso_px)*paso_px,
    x2 + paso_px,
    paso_px)
yticks = np.arange(
    yc + np.floor((y1 - yc)/paso_px)*paso_px,
    y2 + paso_px,
    paso_px)

ax_img.xaxis.set_major_locator(FixedLocator(xticks))
ax_img.yaxis.set_major_locator(FixedLocator(yticks))



ax_img.set_xlabel("x [mm]")
ax_img.set_ylabel("y [mm]")

# if centro_imagen:
#     ax_img.plot(x0,y0,marker="+",color="cyan",markersize=15,mew=3)
  
xmin = gauss["xmin"]
xmax = gauss["xmax"]
ymin = gauss["ymin"]
ymax = gauss["ymax"]

ax_img.axhline(ymin, color="grey", ls="-")
ax_img.axhline(ymax, color="grey", ls="-")

#ax_img.axvline(xmin, color="grey", ls="-")
#ax_img.axvline(xmax, color="grey", ls="-")


plt.show()



#Guardo las intensidades mínimas y máximas para que el colormap de la top_hat
#tenga los mismos colores por intensidad que esta
vmin = np.min(gauss["roi"])
vmax = np.max(gauss["roi"])



#%% Foto con π-Shaper


top = analizar_perfil_modificado(ruta_tiff = "/Users/Mauri/Downloads/después pishaper_6.tiff",
                    half_size = 900, mm = False,
                    x0 = 980, y0 = 560, centro_imagen = False,
                    band_halfwidth = 15,
                    ajuste_gauss = False,
                    centro_ajuste = False,
                    graficar_perfiles = False,
                    estadística = False,
                    angulo_max = 360,
                    angulo_paso = 10,
                    grafico_perfiles_x = False
                    )

#%%
fig = plt.figure(figsize=(10, 7), constrained_layout=True)

plt.rcParams.update({
    "font.size": 18,        # tamaño base
    "axes.titlesize": 22,   # títulos
    "axes.labelsize": 24,   # etiquetas de los ejes
    "xtick.labelsize": 18,  # números del eje x
    "ytick.labelsize": 18,  # números del eje y
    "legend.fontsize": 18,  # leyenda (si hay)
})


#graficar_perfiles:

gs = fig.add_gridspec(
    2, 2,
    width_ratios=[0.25, 4],
    height_ratios=[1, 4],
    wspace=0.1
)

ax_top  = fig.add_subplot(gs[0, 1])
ax_cbar = fig.add_subplot(gs[1, 0])
ax_img  = fig.add_subplot(gs[1, 1])


# Imagen
im = ax_img.imshow(
    top["roi"],
    cmap="inferno",
    origin="lower",
    aspect="auto",
    extent=top["extent"],
    vmin = vmin,
    vmax = vmax
)
ax_img.set_xlabel("x [px]")
ax_img.set_ylabel("y [px]")


# centro del haz dentro de la ROI
xc = top["xc"]
yc = top["yc"]

# centro del ajuste en coordenadas GLOBALES
# if centro_ajuste:
#     xc = int(x0_fit + x1)
#     yc = int(y0_fit + y1)


x1 = top["extent"][0]
x2 = top["extent"][1]
y1 = top["extent"][2]
y2 = top["extent"][3]


#if graficar_perfiles:
# Perfil X
ax_top.plot(top["perfil_x"]["x"], top["perfil_x"]["I"], color="grey")
ax_top.set_xlim(x1, x2)
#ax_top.set_xticks([])
ax_top.tick_params(axis="x", which="both",
                   bottom=False, labelbottom=False)
ax_top.grid(True,which="major",linestyle="--",linewidth=0.8,alpha=0.5)
ax_top.set_ylabel("I(x) [u.a.]", fontsize = 18)


# Colorbar en su propio eje
cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label("Intensidad [u.a.]")
ax_cbar.yaxis.set_label_position("left")
ax_cbar.yaxis.set_ticks_position("left")



# Convertir automáticamente px -> mm en los ejes
pixel_size = top["pixel_size"]




ax_img.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{(x - xc) * pixel_size:.0f}")
)

ax_img.yaxis.set_major_formatter(
    FuncFormatter(lambda y, pos: f"{(y - yc) * pixel_size:.0f}")
)


# ticks cada 1 mm
paso_px = 1 / pixel_size
xticks = np.arange(
    xc + np.floor((x1 - xc)/paso_px)*paso_px,
    x2 + paso_px,
    paso_px)
yticks = np.arange(
    yc + np.floor((y1 - yc)/paso_px)*paso_px,
    y2 + paso_px,
    paso_px)

ax_img.xaxis.set_major_locator(FixedLocator(xticks))
ax_img.yaxis.set_major_locator(FixedLocator(yticks))



ax_img.set_xlabel("x [mm]")
ax_img.set_ylabel("y [mm]")

# if centro_imagen:
#     ax_img.plot(x0,y0,marker="+",color="cyan",markersize=15,mew=3)
  
xmin = top["xmin"]
xmax = top["xmax"]
ymin = top["ymin"]
ymax = top["ymax"]

ax_img.axhline(ymin, color="grey", ls="-")
ax_img.axhline(ymax, color="grey", ls="-")

#ax_img.axvline(xmin, color="grey", ls="-")
#ax_img.axvline(xmax, color="grey", ls="-")


plt.show()






#%%Comparación gráfico


















