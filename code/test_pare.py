"""Prueba enfocada SOLO en la señal PARE: deteccion, confirmacion, parada y reanudar."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

import config
import control
import senales
import vision
from drivers import make_driver

W, H = 320, 240
BAND = config.SIGNAL_BAND
fallos = []


def check(nombre, cond, detalle=""):
    print(f"  {'PASA' if cond else 'FALLA'}  {nombre}" + (f"  [{detalle}]" if detalle else ""))
    if not cond:
        fallos.append(nombre)


def octagono(s, cx, cy):
    return np.array([
        (cx, cy - s), (cx + int(s * .7), cy - int(s * .7)), (cx + s, cy),
        (cx + int(s * .7), cy + int(s * .7)), (cx, cy + s),
        (cx - int(s * .7), cy + int(s * .7)), (cx - s, cy),
        (cx - int(s * .7), cy - int(s * .7))], np.int32)


def escena_pare(s, cy, color=(35, 35, 205)):
    img = np.full((H, W, 3), 150, np.uint8)
    cv2.fillPoly(img, [octagono(s, W // 2, cy)], color)
    return img


def line_dummy():
    h, w = 240, 320
    pts = [(w // 2, h - 20 - i * 20) for i in range(4)]
    cnts = [np.array([[w // 2 - 5, h - 20], [w // 2 + 5, h - 20],
                      [w // 2 + 5, h - 30], [w // 2 - 5, h - 30]], np.int32)]
    return vision.LineResult(pts, cnts, np.zeros((h, w), np.uint8), 0, 0.0, [], 0.0, 0,
                            False, None)


print("=" * 70)
print("1. PARE se detecta como COMPLETO (partial=False) al acercarse")
print("=" * 70)

for s, cy in ((18, 150), (24, 175), (30, 190), (30, 210), (34, 218)):
    r = senales.detect_signal(escena_pare(s, cy), *BAND)
    check(f"octagono s={s} cy={cy} se detecta y NO es parcial",
          r is not None and r.label == "PARE" and not r.partial,
          f"label={r.label if r else None} partial={r.partial if r else '-'}")

print()
print("=" * 70)
print("2. PARE NO se detecta en el aire (fuera de la banda de suelo)")
print("=" * 70)

r = senales.detect_signal(escena_pare(26, 12), *BAND)
check("octagono en el aire se descarta", r is None, f"resultado={r.label if r else None}")

print()
print("=" * 70)
print("3. PARE confirmado tras N frames: el robot queda en PARE")
print("=" * 70)

img = escena_pare(30, 190)
brain = control.Brain(make_driver("pulsos"))
t = 0.0
estado_inicial = brain.state.value
confirmado_en = None
for i in range(12):
    brain.on_frame(line_dummy(), now=t)
    brain.on_signal("PARE", t, partial=False)
    msg, _, _ = brain.next_command()
    if brain.state == control.State.PARE and confirmado_en is None:
        confirmado_en = i
    t += 0.1

check("estado inicial es SIGUIENDO", estado_inicial == "SIGUIENDO", estado_inicial)
# La tercera llamada a on_signal (índice 2) lleva _sig_count a 3 y confirma.
check("PARE confirmado tras los frames exigidos",
      confirmado_en == config.SIGNAL_CONFIRM_FRAMES - 1,
      f"confirmado en indice {confirmado_en}, se exigen {config.SIGNAL_CONFIRM_FRAMES} frames")
check("estado final es PARE", brain.state == control.State.PARE, brain.state.value)

print()
print("=" * 70)
print("4. PARE mantiene la parada STOP_DURATION_S y luego reanuda")
print("=" * 70)

brain2 = control.Brain(make_driver("pulsos"))
t = 0.0
for _ in range(config.SIGNAL_CONFIRM_FRAMES + 1):
    brain2.on_frame(line_dummy(), now=t)
    brain2.on_signal("PARE", t, partial=False)
    t += 0.1
check("detenido tras confirmar", brain2.state == control.State.PARE, brain2.state.value)

brain2.on_frame(line_dummy(), now=t + config.STOP_DURATION_S * 0.4)
check("sigue PARE dentro de STOP_DURATION_S",
      brain2.state == control.State.PARE,
      f"a {config.STOP_DURATION_S * 0.4:.1f}s de {config.STOP_DURATION_S}s")

brain2.on_frame(line_dummy(), now=t + config.STOP_DURATION_S + 0.4)
check("reanuda pasado STOP_DURATION_S",
      brain2.state != control.State.PARE, brain2.state.value)

print()
print("=" * 70)
print("5. Cooldown: tras pasar la señal, PARE se rearma")
print("=" * 70)

brain3 = control.Brain(make_driver("pulsos"))
t = 0.0
for _ in range(config.SIGNAL_CONFIRM_FRAMES + 1):
    brain3.on_frame(line_dummy(), now=t)
    brain3.on_signal("PARE", t, partial=False)
    t += 0.1
check("PARE queda desarmado mientras sigue a la vista",
      brain3._sig_armed["PARE"] is False)

for _ in range(int(config.SIGNAL_COOLDOWN_S * 100) + 20):
    brain3.on_signal(None, t)
    brain3.on_frame(line_dummy(), now=t)
    t += 0.1
check("PARE se rearma tras desaparecer la señal",
      brain3._sig_armed["PARE"] is True)

print()
print("=" * 70)
print(f"RESULTADO: {len(fallos)} fallo(s)")
for f in fallos:
    print(f"  - {f}")
print("=" * 70)
sys.exit(1 if fallos else 0)