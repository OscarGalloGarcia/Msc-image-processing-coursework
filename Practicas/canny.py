"""
Detector de bordes de Canny implementado desde cero.
""" 
import argparse
import cv2
import numpy as np


# utilidades
def a_gris(img):
    """BGR -> escala de grises con los pesos de luminancia ITU-R BT.601."""
    if img.ndim == 2:
        return img.astype(np.float64)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float64)


def convolucion(img, kernel):
    """Convolución 2D """
    kernel = np.flipud(np.fliplr(kernel))        # convolución
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")

    # Vista deslizante: (H, W, kh, kw).
    ventanas = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", ventanas, kernel)


# 1. suavizado
def kernel_gaussiano(tam, sigma):
    """Kernel gaussiano 2D normalizado."""
    k = tam // 2
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


# 2. gradientes
def gradientes_sobel(img):
    """Magnitud y dirección del gradiente con máscaras de Sobel."""
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float64)

    gx = convolucion(img, kx)
    gy = convolucion(img, ky)

    magnitud = np.hypot(gx, gy)
    if magnitud.max() > 0:
        magnitud = magnitud / magnitud.max() * 255.0
    angulo = np.rad2deg(np.arctan2(gy, gx)) % 180.0   # 0–180°
    return magnitud, angulo


# 3. supresión de no-máximos
def supresion_no_maximos(mag, ang):
    """Adelgaza los bordes dejando solo los máximos locales en la dirección
    perpendicular al borde (se cuantiza el ángulo a 0°, 45°, 90°, 135°)."""
    H, W = mag.shape
    salida = np.zeros((H, W), dtype=np.float64)

    m = np.pad(mag, 1, mode="constant")   # padding para no salirse del arreglo

    # Vecinos según el sector angular
    sectores = [
        ((ang < 22.5) | (ang >= 157.5), (0, -1), (0, 1)),     # horizontal  0°
        ((ang >= 22.5) & (ang < 67.5),  (1, -1), (-1, 1)),    # diagonal   45°
        ((ang >= 67.5) & (ang < 112.5), (-1, 0), (1, 0)),     # vertical   90°
        ((ang >= 112.5) & (ang < 157.5), (-1, -1), (1, 1)),   # diagonal  135°
    ]

    ys, xs = np.mgrid[1:H + 1, 1:W + 1]
    for mascara, (dy1, dx1), (dy2, dx2) in sectores:
        v1 = m[ys + dy1, xs + dx1]
        v2 = m[ys + dy2, xs + dx2]
        es_max = (mag >= v1) & (mag >= v2) & mascara
        salida[es_max] = mag[es_max]

    return salida


#  4. doble umbral
DEBIL, FUERTE = 75, 255


def doble_umbral(img, low_ratio, high_ratio):
    """Clasifica píxeles en fuertes, débiles y descartados."""
    alto = img.max() * high_ratio
    bajo = alto * low_ratio if low_ratio < 1 else img.max() * low_ratio

    res = np.zeros_like(img, dtype=np.uint8)
    res[img >= alto] = FUERTE
    res[(img >= bajo) & (img < alto)] = DEBIL
    return res


# 5. histéresis
def histeresis(img):
    """Un píxel débil sobrevive solo si está conectado (8-vecindad) a uno
    fuerte, directa o transitivamente. Se resuelve con BFS iterativa."""
    H, W = img.shape
    salida = np.zeros((H, W), dtype=np.uint8)
    salida[img == FUERTE] = 255

    debiles = (img == DEBIL)
    cambio = True
    while cambio:                       # propagación hasta punto fijo
        p = np.pad(salida, 1, mode="constant")
        vecino_fuerte = np.zeros((H, W), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                vecino_fuerte |= p[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] > 0

        nuevos = debiles & vecino_fuerte & (salida == 0)
        cambio = nuevos.any()
        salida[nuevos] = 255

    return salida


# pipeline
def canny(img, sigma=1.4, tam_kernel=5, low_ratio=0.05, high_ratio=0.15,
          devolver_etapas=False):
    gris = a_gris(img)
    suave = convolucion(gris, kernel_gaussiano(tam_kernel, sigma))
    mag, ang = gradientes_sobel(suave)
    delgado = supresion_no_maximos(mag, ang)
    umbralada = doble_umbral(delgado, low_ratio, high_ratio)
    bordes = histeresis(umbralada)

    if devolver_etapas:
        return bordes, {
            "1_gris": gris,
            "2_suavizado": suave,
            "3_magnitud": mag,
            "4_no_maximos": delgado,
            "5_doble_umbral": umbralada,
            "6_histeresis": bordes,
        }
    return bordes


def main():
    ap = argparse.ArgumentParser(description="Canny desde cero")
    ap.add_argument("imagen", nargs="?", default="data/lena.bmp")
    ap.add_argument("--sigma", type=float, default=1.4)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--low", type=float, default=0.05)
    ap.add_argument("--high", type=float, default=0.15)
    ap.add_argument("--salida", default="bordes.png")
    ap.add_argument("--etapas", action="store_true",
                    help="guarda las imágenes intermedias")
    args = ap.parse_args()

    img = cv2.imread(args.imagen)
    if img is None:
        raise SystemExit(f"No pude leer '{args.imagen}'")

    bordes, etapas = canny(img, args.sigma, args.kernel,
                           args.low, args.high, devolver_etapas=True)

    cv2.imwrite(args.salida, bordes)
    print(f"Guardado: {args.salida}")

    if args.etapas:
        for nombre, etapa in etapas.items():
            e = etapa.astype(np.float64)
            e = (e - e.min()) / (np.ptp(e) + 1e-9) * 255
            cv2.imwrite(f"{nombre}.png", e.astype(np.uint8))
            print(f"Guardado: {nombre}.png")


if __name__ == "__main__":
    main()