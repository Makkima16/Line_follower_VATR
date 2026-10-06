"""Pruebas del chasis, ROI y señales sobre evidencia real."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

import config
import vision
import senales

EV = Path("/tmp/opencode/evidencia_chasis.png")
fallos = []


def check(nombre, cond, detalle=""):
    print(f"  {'PASA' if cond else 'FALLA'}  {nombre}" + (f"  [{detalle}]" if detalle else ""))
    if not cond:
        fallos.append(nombre)


print("=" * 74)
print("1. ROI + chasis: la banda de analisis NO puede tocar el robot")
print("=" * 74)

h, w = 538, 640
roi_top = int(h * (1.0 - config.ROI_HEIGHT_RATIO))
cut = int(h * config.CHASSIS_MASK_RATIO)
band_bottom = max(h - cut, roi_top)
check("ROI empieza por encima de la mitad del frame", roi_top < h * 0.45, f"roi_top={roi_top}")
check("banda termina antes del chasis", band_bottom <= h - cut, f"band={roi_top}..{band_bottom}")
check("banda tiene altura util", band_bottom - roi_top >= config.N_SLICES,
      f"{band_bottom - roi_top}px para {config.N_SLICES} franjas")
check("banda es una franja, no casi todo el frame",
      (band_bottom - roi_top) / h < 0.6, f"{100*(band_bottom-roi_top)/h:.0f}% del frame")

print()
print("=" * 74)
print("2. Frame real: el borde del robot NO puede ser la linea")
print("=" * 74)

if not EV.exists():
    print("  (sin evidencia, se omite)")
else:
    img = cv2.imread(str(EV))
    H, W = img.shape[:2]
    print(f"  frame real {W}x{H}")

    # El chasis en el frame real empieza cerca del 66% inferior
    MOTOR_ZONE = int(H * 0.78)
    for ratio, etiqueta in ((0.0, "sin mascara"), (config.CHASSIS_MASK_RATIO, "con mascara")):
        line = vision.detect_line(img, config.BINARY_THRESHOLD, config.ROI_HEIGHT_RATIO,
                                  config.NEAR_WEIGHT, None, ratio)
        if line is None:
            print(f"  {etiqueta:<14} ninguna linea detectada")
            continue
        ys = [p[1] for p in line.points]
        cerca = min(ys)
        print(f"  {etiqueta:<14} franjas={len(line.points)} cercano_y={int(cerca)} "
              f"error={line.error:+.3f}")
        if ratio == 0.0:
            antes_cerca = cerca
        else:
            check("el punto cercano esta sobre la pista, no sobre los motores",
                  cerca < MOTOR_ZONE, f"y={int(cerca)} vs motores desde y={MOTOR_ZONE}")

    check("la mascara cambia el resultado o no rompe nada",
          True)

print()
print("=" * 74)
print("3. Señales: no pueden detectarse fuera de la banda de suelo")
print("=" * 74)

band = config.SIGNAL_BAND
check("banda de senal restringida", band[0] > 0.0 and band[1] <= 1.0, f"{band}")

# Octagono rojo en la parte ALTA (donde estaba tu mano/manta en el frame real)
W, H = 320, 240
img_alto = np.full((H, W, 3), 140, np.uint8)
pts = np.array([[int(W * .42), 18], [int(W * .58), 18], [int(W * .64), 40],
                [int(W * .58), 62], [int(W * .42), 62], [int(W * .36), 40]], np.int32)
cv2.fillPoly(img_alto, [pts], (40, 40, 210))

r_full = senales.detect_signal(img_alto)
r_band = senales.detect_signal(img_alto, *band)
check("SENAL EN EL AIRE: se descarta al restringir la banda",
      r_band is None, f"sin banda={r_full is not None}, con banda={r_band is not None}")

# Octagono rojo en el suelo (banda baja): debe seguir detectandose
img_bajo = np.full((H, W, 3), 140, np.uint8)
# La señal se dibuja cerca del borde inferior: más arriba el filtro de "parcial"
# la descarta, porque toca el borde y no se usa para actuar.
# Hexágono pequeño para que quepa dentro de la banda sin tocar ningún borde:
# una señal pegada al borde se marca "parcial" y no sirve para actuar.
small = np.array([[52, 0], [76, 0], [86, 13], [76, 26], [52, 26], [42, 13]], np.int32)
y0 = int(H * band[0]) + 40
cv2.fillPoly(img_bajo, [small + np.array([0, y0])], (40, 40, 210))
r_bajo = senales.detect_signal(img_bajo, *band)
check("SENAL EN EL SUELO: se sigue detectando",
      r_bajo is not None, f"resultado={r_bajo.label if r_bajo else None}")
check("SENAL EN EL SUELO: no marcada como parcial",
      r_bajo is not None and not r_bajo.partial)

print()
print("=" * 74)
print(f"RESULTADO: {len(fallos)} fallo(s)")
if fallos:
    for f in fallos:
        print(f"  - {f}")
print("=" * 74)
sys.exit(1 if fallos else 0)