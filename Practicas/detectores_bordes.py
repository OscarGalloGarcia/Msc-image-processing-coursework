"""
Detectores de Beaudet y de Wang-Brady implementados desde cero y aplicados
sobre video (cámara o archivo). Se comparan cuadro a cuadro.

Beaudet (1978) – determinante del Hessiano:
    DET = Ixx * Iyy - Ixy^2
    Respuestas grandes en |DET| marcan puntos donde la intensidad tiene
    curvatura fuerte en las dos direcciones (esquinas, uniones, manchas).

Wang-Brady (1995) – curvatura a lo largo del borde:
    Itt = (Ix^2 Iyy - 2 Ix Iy Ixy + Iy^2 Ixx) / |∇I|^2   (2a derivada tangencial)
    Γ   = Itt^2 - k |∇I|^2
    Un punto es esquina si Γ > umbral y además |∇I|^2 es suficientemente
    grande (está sobre un borde). El término -k|∇I|^2 castiga los bordes rectos.

Uso:
    python Practicas/detectores_bordes.py              # cámara 0
    python Practicas/detectores_bordes.py video.mp4    # archivo
Teclas: q salir, p pausa, s guardar captura.
Al terminar se guardan metricas.csv y comparacion.png en --salida.
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

from canny import a_gris, convolucion, kernel_gaussiano


# derivadas (kernels de convolución, x hacia la derecha, y hacia abajo)
KX = np.array([[1, 0, -1]], dtype=np.float64) / 2.0
KY = KX.T
KXX = np.array([[1, -2, 1]], dtype=np.float64)
KYY = KXX.T
KXY = np.array([[1, 0, -1],
                [0, 0, 0],
                [-1, 0, 1]], dtype=np.float64) / 4.0


def derivadas(img, sigma, tam_kernel):
    """Suaviza y devuelve Ix, Iy, Ixx, Iyy, Ixy."""
    suave = convolucion(img, kernel_gaussiano(tam_kernel, sigma))
    return (convolucion(suave, KX), convolucion(suave, KY),
            convolucion(suave, KXX), convolucion(suave, KYY),
            convolucion(suave, KXY))


# detectores
def respuesta_beaudet(d):
    _, _, ixx, iyy, ixy = d
    return np.abs(ixx * iyy - ixy ** 2)


def respuesta_wang(d, k):
    ix, iy, ixx, iyy, ixy = d
    grad2 = ix ** 2 + iy ** 2
    itt = (ix ** 2 * iyy - 2 * ix * iy * ixy + iy ** 2 * ixx) / (grad2 + 1e-9)
    return itt ** 2 - k * grad2, grad2


# selección de puntos
def maximos_locales(resp, radio):
    """Máscara de píxeles que son máximo en su vecindad (2r+1)x(2r+1)."""
    tam = 2 * radio + 1
    p = np.pad(resp, radio, mode="constant", constant_values=-np.inf)
    vmax = np.lib.stride_tricks.sliding_window_view(p, (tam, tam)).max(axis=(2, 3))
    return resp >= vmax


def detectar(resp, umbral_rel, radio, max_puntos, mascara_extra=None):
    """Umbral relativo al máximo + supresión de no-máximos.
    Devuelve arreglo (N, 2) de coordenadas (y, x) ordenadas por respuesta."""
    tope = resp.max()
    if tope <= 0:
        return np.empty((0, 2), dtype=int)
    m = (resp > umbral_rel * tope) & maximos_locales(resp, radio)
    if mascara_extra is not None:
        m &= mascara_extra
    ys, xs = np.nonzero(m)
    orden = np.argsort(resp[ys, xs])[::-1][:max_puntos]
    return np.stack([ys[orden], xs[orden]], axis=1)


# métricas de comparación
def fraccion_emparejada(a, b, tol):
    """Fracción de puntos de `a` que tienen un punto de `b` a <= tol px."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float((dist.min(axis=1) <= tol).mean())


# visualización
def dibujar(frame, puntos, color, titulo, info):
    out = frame.copy()
    for y, x in puntos:
        cv2.circle(out, (int(x), int(y)), 4, color, 1, cv2.LINE_AA)
    cv2.putText(out, titulo, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(out, info, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def mapa_calor(resp):
    r = np.log1p(np.maximum(resp, 0))           # compresión logarítmica
    r = (r / (r.max() + 1e-9) * 255).astype(np.uint8)
    return cv2.applyColorMap(r, cv2.COLORMAP_INFERNO)


def graficar(filas, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = np.array([r["frame"] for r in filas])
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(f, [r["n_beaudet"] for r in filas], label="Beaudet")
    ax[0].plot(f, [r["n_wang"] for r in filas], label="Wang-Brady")
    ax[0].set_ylabel("puntos detectados")
    ax[1].plot(f, [r["ms_beaudet"] for r in filas], label="Beaudet")
    ax[1].plot(f, [r["ms_wang"] for r in filas], label="Wang-Brady")
    ax[1].set_ylabel("tiempo (ms)")
    ax[2].plot(f, [r["rep_beaudet"] for r in filas], label="Beaudet")
    ax[2].plot(f, [r["rep_wang"] for r in filas], label="Wang-Brady")
    ax[2].plot(f, [r["coincidencia"] for r in filas], "--", label="Beaudet∩Wang")
    ax[2].set_ylabel("fracción")
    ax[2].set_xlabel("frame")
    ax[2].set_ylim(0, 1.05)
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(loc="upper right")
    ax[0].set_title("Comparación Beaudet vs Wang-Brady por frame")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def resumen(filas):
    print("\nResumen (promedio por frame)")
    print(f"{'':16}{'Beaudet':>12}{'Wang-Brady':>12}")
    for nombre, kb, kw in [("puntos", "n_beaudet", "n_wang"),
                           ("tiempo ms", "ms_beaudet", "ms_wang"),
                           ("repetibilidad", "rep_beaudet", "rep_wang")]:
        b = np.mean([r[kb] for r in filas])
        w = np.mean([r[kw] for r in filas])
        print(f"{nombre:16}{b:12.3f}{w:12.3f}")
    print(f"{'coincidencia':16}{np.mean([r['coincidencia'] for r in filas]):12.3f}")


def main():
    ap = argparse.ArgumentParser(description="Beaudet vs Wang-Brady en video")
    ap.add_argument("fuente", nargs="?", default="0",
                    help="índice de cámara o ruta de video")
    ap.add_argument("--ancho", type=int, default=480,
                    help="se redimensiona cada frame a este ancho")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--umbral-beaudet", type=float, default=0.05,
                    help="fracción del máximo de |DET|")
    ap.add_argument("--umbral-wang", type=float, default=0.05,
                    help="fracción del máximo de Γ")
    ap.add_argument("--k", type=float, default=0.04,
                    help="peso del término de gradiente en Wang-Brady")
    ap.add_argument("--umbral-grad", type=float, default=0.02,
                    help="|∇I|^2 mínimo (fracción del máximo) para Wang-Brady")
    ap.add_argument("--radio", type=int, default=4, help="radio de NMS")
    ap.add_argument("--max-puntos", type=int, default=300)
    ap.add_argument("--tol", type=float, default=3.0,
                    help="distancia (px) para considerar dos puntos iguales")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = sin límite")
    ap.add_argument("--salida", default="resultados_detectores")
    ap.add_argument("--sin-ventana", action="store_true",
                    help="no mostrar video (solo métricas)")
    args = ap.parse_args()

    fuente = int(args.fuente) if args.fuente.isdigit() else args.fuente
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir '{args.fuente}'")
    os.makedirs(args.salida, exist_ok=True)

    filas = []
    prev_b = prev_w = None
    n = 0
    pausa = False
    while True:
        if not pausa:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (args.ancho, int(h * args.ancho / w)))
            gris = a_gris(frame)

            t0 = time.perf_counter()
            d = derivadas(gris, args.sigma, args.kernel)
            t_deriv = time.perf_counter() - t0

            t0 = time.perf_counter()
            rb = respuesta_beaudet(d)
            pb = detectar(rb, args.umbral_beaudet, args.radio, args.max_puntos)
            ms_b = (time.perf_counter() - t0 + t_deriv) * 1000

            t0 = time.perf_counter()
            rw, grad2 = respuesta_wang(d, args.k)
            sobre_borde = grad2 > args.umbral_grad * grad2.max()
            pw = detectar(rw, args.umbral_wang, args.radio, args.max_puntos,
                          sobre_borde)
            ms_w = (time.perf_counter() - t0 + t_deriv) * 1000

            fila = {
                "frame": n,
                "n_beaudet": len(pb),
                "n_wang": len(pw),
                "ms_beaudet": ms_b,
                "ms_wang": ms_w,
                # puntos que reaparecen respecto al frame anterior
                "rep_beaudet": fraccion_emparejada(pb, prev_b, args.tol) if prev_b is not None else np.nan,
                "rep_wang": fraccion_emparejada(pw, prev_w, args.tol) if prev_w is not None else np.nan,
                # cuántos puntos de Beaudet también encontró Wang
                "coincidencia": fraccion_emparejada(pb, pw, args.tol),
            }
            filas.append(fila)
            prev_b, prev_w = pb, pw
            n += 1

            vb = dibujar(frame, pb, (0, 0, 255), "Beaudet",
                         f"{len(pb)} pts  {ms_b:.1f} ms")
            vw = dibujar(frame, pw, (0, 255, 0), "Wang-Brady",
                         f"{len(pw)} pts  {ms_w:.1f} ms")
            mosaico = np.vstack([np.hstack([vb, vw]),
                                 np.hstack([mapa_calor(rb), mapa_calor(rw)])])

        if not args.sin_ventana:
            cv2.imshow("Beaudet | Wang-Brady", mosaico)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("p"):
                pausa = not pausa
            if tecla == ord("s"):
                ruta = os.path.join(args.salida, f"captura_{n:05d}.png")
                cv2.imwrite(ruta, mosaico)
                print(f"Guardado: {ruta}")
        if args.max_frames and n >= args.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    if not filas:
        raise SystemExit("No se procesó ningún frame")

    cv2.imwrite(os.path.join(args.salida, "ultimo_frame.png"), mosaico)
    ruta_csv = os.path.join(args.salida, "metricas.csv")
    with open(ruta_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)
    # el primer frame no tiene repetibilidad
    filas_rep = filas[1:] if len(filas) > 1 else filas
    graficar(filas_rep, os.path.join(args.salida, "comparacion.png"))
    resumen(filas_rep)
    print(f"\nGuardado: {ruta_csv}, comparacion.png, ultimo_frame.png en {args.salida}/")


if __name__ == "__main__":
    main()
