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


def hist_acumulado(Hist):
    Hist_acum = np.cumsum(Hist)
    Hist_acum = Hist_acum / Hist_acum[-1]

    return Hist_acum


# Histograma especificado para canal rojo
def hist_R(x):
    return np.abs(x - 128)


# Histograma especificado para canal verde
def hist_G(x):
    return 255 - x


# Histograma especificado para canal azul
def hist_B(x):
    return np.ones(256) + 100


def especificacion(I_in, Hist_ref):

    # Histograma original del canal
    Hist_org = cv2.calcHist(
        [I_in],
        [0],
        None,
        [256],
        [0,256]
    )

    # Histogramas acumulados
    Hist_org_acum = hist_acumulado(Hist_org)
    Hist_ref_acum = hist_acumulado(Hist_ref)

    # Transformacion
    T = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        diferencia = np.abs(
            Hist_ref_acum - Hist_org_acum[i]
        )

        T[i] = np.argmin(diferencia)

    # Aplicar transformacion al canal
    I_out = T[I_in]

    return I_out


if __name__ == "__main__":

    I_org = cv2.imread("data/lena.bmp") # Lee la imagen


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
    plt.xlabel("Nivel de intensidad")
    plt.ylabel("Numero de pixeles")
    plt.xlim([0,256])

    plt.show()


    #3 Histogramas por especificacion:
    x = np.arange(256)

    Hist_R = hist_R(x)
    Hist_G = hist_G(x)
    Hist_B = hist_B(x)

    plt.figure()

    plt.plot(Hist_R, color='r')
    plt.plot(Hist_G, color='g')
    plt.plot(Hist_B, color='b')

    plt.title("Histogramas por especificacion")
    plt.xlabel("Nivel de intensidad")
    plt.ylabel("Valor de la funcion")
    plt.xlim([0,256])

    plt.show()


    # Separar los canales de la imagen
    # OpenCV utiliza orden BGR
    B, G, R = cv2.split(I_org)


    # Especificacion de cada canal
    R_out = especificacion(
        I_in=R,
        Hist_ref=Hist_R
    )

    G_out = especificacion(
        I_in=G,
        Hist_ref=Hist_G
    )

    B_out = especificacion(
        I_in=B,
        Hist_ref=Hist_B
    )


    #4 Imagen resultante:
    I_out = cv2.merge([
        B_out,
        G_out,
        R_out
    ])

    cv2.imshow("Imagen resultante", I_out)
    cv2.waitKey(100)

    cv2.imwrite("data/I_out.bmp", I_out)


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

    plt.title("Histograma de la imagen resultante")
    plt.xlabel("Nivel de intensidad")
    plt.ylabel("Numero de pixeles")
    plt.xlim([0,256])

    plt.savefig("data/histograma_resultante.png")

    plt.show()


    cv2.waitKey(0)
    cv2.destroyAllWindows()