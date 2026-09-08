# Oscar Alberto Gallo García
# Maestría en Ciencias en Robótica e Inteligencia Artificial
# 
# Actividad: Calcular el histograma de una imagen.
# Descripcion: 
# 1) I_org -> Imagen original de entrada.
# 2) Hist_org -> Histograma de la imagen original.
# 3) Histogramas por especificacion definidos por funciones.
#    Definir uno para cada canal RGB
# 4) I_out -> Mostrar imagen resultante
# 5) Hist_out -> Mostrar el histograma de la imagen resultante.

import numpy as np
import cv2 
from matplotlib import pyplot as plt


def histograma(I_in, color):
    histogramas = []

    for i, col in enumerate(color):
        hist = cv2.calcHist([I_in], [i], None, [256], [0,256])
        histogramas.append(hist)

    return histogramas


def plot_hist(Hist, color):
    for hist, col in zip(Hist, color):
        plt.plot(hist, color=col)


def hist_acumulado(hist):
    hist_acum = np.cumsum(hist)
    hist_acum = hist_acum / hist_acum[-1]

    return hist_acum


def hist_especificado(media, sigma):
    x = np.arange(256)

    hist = np.exp(
        -((x - media) ** 2) / (2 * sigma ** 2)
    )

    return hist


def especificacion(canal, Hist_ref):

    # Histograma del canal original
    Hist_org = cv2.calcHist(
        [canal], [0], None, [256], [0,256]
    )

    # Histogramas acumulados
    Hist_org_acum = hist_acumulado(Hist_org)
    Hist_ref_acum = hist_acumulado(Hist_ref)

    # Tabla de transformacion
    T = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        diferencia = np.abs(
            Hist_ref_acum - Hist_org_acum[i]
        )

        T[i] = np.argmin(diferencia)

    # Aplicar transformacion
    canal_out = T[canal]

    return canal_out


if __name__ == "__main__":

    I_org = cv2.imread("data/lena.bmp") #Lee la imagen
    #I_org = cv2.cvtColor(I_org, cv2.COLOR_BGR2RGB) # Cambia el formato de canales de color


    #1 Imagen original:
    cv2.imshow("Imagen original", I_org)
    cv2.waitKey(100)


    #2 Histograma de la imagen original:
    color = ('b','g','r')

    Hist_org = histograma(
        I_in=I_org,
        color=color
    )

    plt.figure()
    plot_hist(
        Hist=Hist_org,
        color=color
    )

    plt.title("Histograma original")
    plt.xlabel("Intensidad")
    plt.ylabel("Numero de pixeles")
    plt.xlim([0,256])

    plt.show()


    #3 Histogramas por especificacion:
    Hist_B = hist_especificado(
        media=80,
        sigma=40
    )

    Hist_G = hist_especificado(
        media=130,
        sigma=40
    )

    Hist_R = hist_especificado(
        media=180,
        sigma=40
    )

    Hist_ref = [
        Hist_B,
        Hist_G,
        Hist_R
    ]

    plt.figure()

    plot_hist(
        Hist=Hist_ref,
        color=color
    )

    plt.title("Histogramas por especificacion")
    plt.xlabel("Intensidad")

    plt.show()


    # Separar canales de la imagen:
    B, G, R = cv2.split(I_org)


    # Aplicar especificacion:
    B_out = especificacion(
        canal=B,
        Hist_ref=Hist_B
    )

    G_out = especificacion(
        canal=G,
        Hist_ref=Hist_G
    )

    R_out = especificacion(
        canal=R,
        Hist_ref=Hist_R
    )


    #4 Imagen resultante:
    I_out = cv2.merge([
        B_out,
        G_out,
        R_out
    ])

    cv2.imshow(
        "Imagen resultante",
        I_out
    )

    cv2.waitKey(100)


    #5 Histograma de la imagen resultante:
    Hist_out = histograma(
        I_in=I_out,
        color=color
    )

    plt.figure()

    plot_hist(
        Hist=Hist_out,
        color=color
    )

    plt.title("Histograma resultante")
    plt.xlabel("Intensidad")
    plt.ylabel("Numero de pixeles")
    plt.xlim([0,256])

    plt.show()


    cv2.waitKey(0)
    cv2.destroyAllWindows()