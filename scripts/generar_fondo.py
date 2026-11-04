"""Genera la clase 4 ("fondo": pista sin señal) recortando las fotos de pistas/.

Cada recorte tiene proporción 3:4 (como las fotos del dataset) y se descarta si
contiene píxeles rojos o verdes saturados, es decir, si apareció una señal.

Uso (desde la raíz del proyecto):  .venv/bin/python scripts/generar_fondo.py
"""
import os
import random

import cv2
import numpy as np

PISTAS = ["zigzag.jpeg", "final.jpeg", "prueba.jpeg", "cintas.jpeg", "signals.jpeg", "ramificacion.png"]
SALIDA = os.path.join("dataset", "4")
POR_FOTO = 17
FRAC_COLOR_MAX = 0.0005   # fracción máxima de píxeles rojo/verde saturados
SEMILLA = 42


def hay_senal(recorte):
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    rojo = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
    verde = cv2.inRange(hsv, (45, 60, 60), (90, 255, 255))
    return np.count_nonzero(rojo | verde) / rojo.size > FRAC_COLOR_MAX


def main():
    random.seed(SEMILLA)
    os.makedirs(SALIDA, exist_ok=True)
    total = 0
    for nombre in PISTAS:
        img = cv2.imread(os.path.join("pistas", nombre))
        h, w = img.shape[:2]
        guardados, intentos = 0, 0
        while guardados < POR_FOTO and intentos < 500:
            intentos += 1
            alto = int(h * random.uniform(0.5, 1.0))
            ancho = int(alto * 3 / 4)
            if ancho > w:
                continue
            y, x = random.randint(0, h - alto), random.randint(0, w - ancho)
            rec = img[y:y + alto, x:x + ancho]
            if hay_senal(rec):
                continue
            rec = cv2.resize(rec, (576, 768), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(SALIDA, f"fondo_{os.path.splitext(nombre)[0]}_{guardados:02d}.jpg"), rec)
            guardados += 1
        print(f"{nombre}: {guardados} recortes")
        total += guardados
    print(f"Total en {SALIDA}: {total}")


if __name__ == "__main__":
    main()
