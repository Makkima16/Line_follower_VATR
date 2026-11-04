"""Pruebas del chasis, ROI y señales sobre evidencia real."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

import config
import vision
import senales

EV = Path(__file__).parent.parent / "pistas" / "chasis_real.png"   # frame real del celular montado
fallos = []


def check(nombre, cond, detalle=""):
    print(f"  {'PASA' if cond else 'FALLA'}  {nombre}" + (f"  [{detalle}]" if detalle else ""))
    if not cond:
        fallos.append(nombre)


print("=" * 74)
print("1. ROI + chasis: la banda de analisis NO puede tocar el robot")
print("=" * 74)

h, w = 240, 320
roi_bottom = int(h * config.ROI_HEIGHT_RATIO)          # La ROI es la parte SUPERIOR
chassis_top = h - int(h * config.CHASSIS_MASK_RATIO)   # Desde aquí hacia abajo es robot
band_bottom = min(roi_bottom, chassis_top)
check("la ROI termina por encima del chasis", band_bottom <= chassis_top,
      f"ROI=0..{band_bottom}, chasis desde {chassis_top}")
check("banda tiene altura util", band_bottom >= 10 * config.N_SLICES,
      f"{band_bottom}px para {config.N_SLICES} franjas")

# Línea recta + cuerpo oscuro abajo: una sola salida (arriba). Si las filas del
# chasis se pintaran como línea saldrían 3 (ramificación fantasma).
syn = np.full((h, w, 3), 200, np.uint8)
cv2.line(syn, (160, 0), (160, h), (20, 20, 20), 22)
cv2.rectangle(syn, (40, int(h * 0.70)), (280, h), (30, 30, 30), -1)
line = vision.detect_line(syn, config.BINARY_THRESHOLD)
check("linea recta con chasis: exactamente 1 salida", line is not None and len(line.exits) == 1,
      f"salidas={[e.px for e in line.exits] if line else None}")
check("linea recta con chasis: error ~ 0", line is not None and abs(line.error) < 0.05)
# Trackbars cruzados (ROI minima y chasis maximo) no pueden romper la deteccion
try:
    vision.detect_line(syn, config.BINARY_THRESHOLD, 0.10, 0.6, None, 0.80)
    extremos_ok = True
except Exception as e:
    extremos_ok = False
check("trackbars extremos no lanzan error", extremos_ok)

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

    # En el frame real el robot (sensor ultrasónico y soportes) empieza en ~70% del alto
    ROBOT_TOP = int(H * 0.70)
    line = vision.detect_line(img, config.BINARY_THRESHOLD)
    check("se detecta la cinta", line is not None)
    if line is not None:
        ys = [p[1] for p in line.points]
        print(f"  franjas={len(line.points)} y={int(min(ys))}..{int(max(ys))} error={line.error:+.3f} "
              f"salidas={[e.px for e in line.exits]}")
        check("ningun punto de la linea cae sobre el robot", max(ys) < ROBOT_TOP,
              f"y_max={int(max(ys))} vs robot desde y={ROBOT_TOP}")
        check("la cinta se sigue en todas las franjas", len(line.points) == config.N_SLICES)
        check("una sola salida (sin ramificacion fantasma)", len(line.exits) == 1)
        check("error pequeno: la cinta esta casi centrada", abs(line.error) < 0.3)

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