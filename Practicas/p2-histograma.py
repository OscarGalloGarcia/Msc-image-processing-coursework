#Actividad: Calcular el histograma de una imagen.

# 1) I_org -> Imagen original de entrada.
# 2) Hist_org -> Histograma de la imagen original.
# 3) Histogramas por especificacion definidos por funciones.
#    Definir uno para cada canal RGB
# 4) I_out -> Mostrar imagen resultante
# 5) Hist_out -> Mostrar el histograma de la imagen resultante.

import numpy as np
import cv2 
from matplotlib import pyplot as plt

I_org = cv2.imread("data/lena.bmp") #Lee la imagen
#I_org = cv2.cvtColor(I_org, cv2.COLOR_BGR2RGB) # Cambia el formato de canales de color

#1
cv2.imshow("Imagen original", I_org)
cv2.waitKey(100)

#2 Calcular histograma:
color = ('b,g,r')
#for i, col in enumerate(color):
#    Hist_org = cv2.calcHist([I_org], [i], None, [256], [0,256])
#    plt.plot(Hist_org, color=col)
#    plt.xlim([0,256])
#plt.draw()
#plt.pause(100)
#

def histograma(I_in, color):
    for i, col in enumerate(color):
        hist = cv2.calcHist([I_in], [i], None, [256], [0,256])

    return hist

def hist_acumulado(I_in):
    pass