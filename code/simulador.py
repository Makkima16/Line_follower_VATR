#!/usr/bin/env python3
"""
Simulador de lazo cerrado: prueba el seguidor completo sin robot.

Una foto cenital de la pista hace de "piso". Sobre ella se mueve un robot
virtual y, en cada frame, se recorta lo que vería su cámara:

    pose (x, y, θ) → cámara virtual → vision.detect_line → Brain (PD)
         ↑                                                     │
         └──── cinemática uniciclo ← driver.next() ←───────────┘

A diferencia de mover el celular con la mano, aquí el giro que ordena el
control SÍ cambia lo que ve la cámara, así que la trayectoria dibujada es la
que haría el robot real (con el modelo aproximado de config.SIM_*).

Cámara virtual: el celular mira hacia adelante y abajo, así que ve un
trapecio de piso (angosto cerca, ancho lejos). Con cv2.getPerspectiveTransform
se calcula la homografía que lleva las 4 esquinas de ese trapecio (en píxeles
del mapa) a las 4 esquinas del frame, y cv2.warpPerspective lo "fotografía".

Se usa un reloj simulado, no el del PC: los resultados no dependen de lo
rápido que corra la máquina y se pueden repetir.

Uso:
  python simulador.py ../pistas/zigzag.jpeg
  python simulador.py pista.jpg --ancho-cm 200 --driver velocidades

Teclas: q = salir, espacio = pausa, r = reiniciar, + / - = más/menos rápido.
Ratón (ventana "Mapa"): clic = posición inicial, segundo clic = hacia dónde mira.
"""

import argparse
import math
from collections import deque

import cv2
import numpy as np

import config
import vision
from control import Brain, State
from drivers import DRIVERS, make_driver
from main import WIN_TUNE, create_trackbars, draw_overlay, read_trackbars

WIN_MAP = "Mapa"
WIN_CAM = "Camara virtual"
MAP_VIEW_H = 720          # Alto de la ventana del mapa (px de pantalla)
STEP_S = 0.005            # Paso de integración de la cinemática


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRÍA
# ─────────────────────────────────────────────────────────────────────────────
def axes(theta):
    """
    Vectores adelante/izquierda en píxeles del mapa (y crece hacia abajo).
    θ se mide antihorario desde +x, como el ω que reportan los drivers.
    """
    fwd = np.array([math.cos(theta), -math.sin(theta)])
    left = np.array([-math.sin(theta), -math.cos(theta)])
    return fwd, left


def camera_footprint(pos, theta, px_per_cm):
    """Esquinas del trapecio visto: lejos-izq, lejos-der, cerca-der, cerca-izq."""
    fwd, left = axes(theta)
    near, far = config.SIM_CAM_NEAR_CM, config.SIM_CAM_NEAR_CM + config.SIM_CAM_DEPTH_CM
    nw, fw = config.SIM_CAM_NEAR_WIDTH_CM / 2, config.SIM_CAM_FAR_WIDTH_CM / 2
    pts_cm = [(far, fw), (far, -fw), (near, -nw), (near, nw)]
    return np.float32([pos + (f * fwd + l * left) * px_per_cm for f, l in pts_cm])


class VirtualCamera:
    def __init__(self, track, px_per_cm):
        self.track = track
        self.px_per_cm = px_per_cm
        self.w, self.h = config.PROCESS_WIDTH, config.SIM_CAM_HEIGHT_PX
        self.dst = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]])
        # Fuera del mapa se ve "papel": el color mediano de la foto
        self.paper = tuple(int(c) for c in np.median(track.reshape(-1, 3), axis=0))

    def capture(self, pos, theta):
        src = camera_footprint(pos, theta, self.px_per_cm)
        m = cv2.getPerspectiveTransform(src, self.dst)
        return cv2.warpPerspective(self.track, m, (self.w, self.h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=self.paper)


def auto_start(track, px_per_cm):
    """
    Pose inicial automática: el extremo inferior de la línea, mirando hacia
    donde avanza.

    La línea es el contorno oscuro de mayor área que no toca los bordes
    laterales (así se descartan el espiral del cuaderno o sombras del borde).
    Su punto más bajo es el inicio y la dirección es la del promedio de los
    puntos del contorno cercanos a él.
    """
    mask = vision.line_mask(track, config.BINARY_THRESHOLD)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = track.shape[:2]
    if not cnts:
        return np.array([w / 2, h * 0.9]), math.pi / 2

    def touches_side(c):
        x, _, cw, _ = cv2.boundingRect(c)
        return x <= 2 or x + cw >= w - 2

    inner = [c for c in cnts if not touches_side(c)] or cnts
    pts = max(inner, key=cv2.contourArea).reshape(-1, 2).astype(float)
    tip = pts[np.argmax(pts[:, 1])]
    near = pts[np.linalg.norm(pts - tip, axis=1) < 8 * px_per_cm]
    d = near.mean(axis=0) - tip
    theta = math.atan2(-d[1], d[0]) if np.hypot(*d) > 1 else math.pi / 2
    # Retroceder para que la punta de la línea entre por abajo del frame
    fwd, _ = axes(theta)
    pos = tip - fwd * (config.SIM_CAM_NEAR_CM + 3) * px_per_cm
    return pos, theta


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────
class Simulation:
    def __init__(self, track, px_per_cm, driver_name, start):
        self.cam = VirtualCamera(track, px_per_cm)
        self.px_per_cm = px_per_cm
        self.driver_name = driver_name
        self.start = start
        self.reset()

    def reset(self, start=None):
        if start is not None:
            self.start = start
        self.pos = np.array(self.start[0], float)
        self.theta = self.start[1]
        self.brain = Brain(make_driver(self.driver_name))
        self.t = 0.0
        self.next_frame_t = 0.0
        self.next_cmd_t = 0.0
        self.motion = (0.0, 0.0)          # (v cm/s, ω °/s) del comando en curso
        self.motion_until = 0.0
        self.pending = deque()            # Detecciones esperando la latencia
        self.prev_x = None
        self.last_msg = None
        self.trail = [self.pos.copy()]
        self.distance_cm = 0.0
        self.losses = 0
        self._was_found = True

    def _integrate(self, h):
        v, w_deg = self.motion if self.t < self.motion_until else (0.0, 0.0)
        if v or w_deg:
            fwd, _ = axes(self.theta)
            self.pos += fwd * v * self.px_per_cm * h
            self.theta += math.radians(w_deg) * h
            self.distance_cm += abs(v) * h
        self.t += h

    def advance_one_frame(self, threshold, roi_ratio, near_weight):
        """Corre la física hasta la próxima captura y devuelve (frame, línea)."""
        while True:
            # Hilo de envío: el driver decide el comando y cuánto dura
            if self.t >= self.next_cmd_t:
                msg, wait, (v, w_deg, dur) = self.brain.next_command()
                self.motion, self.motion_until = (v, w_deg), self.t + dur
                self.next_cmd_t = self.t + wait
                if msg is not None:
                    self.last_msg = msg

            # La decisión llega al Brain con retraso (WiFi + procesamiento)
            while self.pending and self.pending[0][0] <= self.t:
                _, line = self.pending.popleft()
                self.brain.on_frame(line, now=self.t)
                if self._was_found and line is None:
                    self.losses += 1
                self._was_found = line is not None

            self._integrate(STEP_S)

            if self.t >= self.next_frame_t:
                self.next_frame_t += 1.0 / config.SIM_CAM_FPS
                frame = self.cam.capture(self.pos, self.theta)
                line = vision.detect_line(frame, threshold, roi_ratio, near_weight, self.prev_x)
                self.prev_x = line.points[0][0] if line is not None else None
                self.pending.append((self.t + config.SIM_LATENCY_S, line))
                if np.linalg.norm(self.pos - self.trail[-1]) > 2:
                    self.trail.append(self.pos.copy())
                return frame, line


# ─────────────────────────────────────────────────────────────────────────────
#  DIBUJO
# ─────────────────────────────────────────────────────────────────────────────
STATE_COLORS = {State.SIGUIENDO: (0, 200, 0), State.BUSCANDO: (0, 140, 255),
                State.DETENIDO: (0, 0, 255)}


def draw_map(track, sim, scale, speed, clicked):
    view = cv2.resize(track, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    to_px = lambda p: (int(p[0] * scale), int(p[1] * scale))

    if len(sim.trail) > 1:
        pts = np.array([to_px(p) for p in sim.trail + [sim.pos]], np.int32)
        cv2.polylines(view, [pts], False, (255, 120, 0), 2, cv2.LINE_AA)

    fp = camera_footprint(sim.pos, sim.theta, sim.px_per_cm) * scale
    cv2.polylines(view, [fp.astype(np.int32)], True, (0, 200, 255), 1, cv2.LINE_AA)

    # Robot: triángulo del tamaño aproximado de un mBot (17 x 13 cm)
    fwd, left = axes(sim.theta)
    k = sim.px_per_cm * scale
    c = sim.pos * scale
    tri = np.array([c + fwd * 9 * k, c - fwd * 8 * k + left * 6.5 * k,
                    c - fwd * 8 * k - left * 6.5 * k], np.int32)
    snap = sim.brain.snapshot()
    cv2.fillPoly(view, [tri], STATE_COLORS[snap["state"]])

    if clicked is not None:
        cv2.circle(view, to_px(clicked), 6, (255, 0, 255), 2)

    lines = [f"t={sim.t:5.1f}s  x{speed:g}  dist={sim.distance_cm / 100:4.2f} m",
             f"{'PAUSA' if snap['paused'] else snap['state'].value}  "
             f"u={snap['u']:+.2f}  perdidas={sim.losses}"]
    for i, text in enumerate(lines):
        cv2.putText(view, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return view


# ─────────────────────────────────────────────────────────────────────────────
#  PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Simulador de lazo cerrado del seguidor")
    p.add_argument("mapa", help="Foto cenital de la pista (línea oscura sobre fondo claro)")
    p.add_argument("--ancho-cm", type=float, default=config.SIM_MAP_WIDTH_CM,
                   help="Ancho real que representa la imagen (escala del mapa)")
    p.add_argument("--driver", choices=list(DRIVERS), default=config.DRIVER)
    return p.parse_args()


def main():
    args = parse_args()
    track = cv2.imread(args.mapa)
    if track is None:
        raise SystemExit(f"No se pudo leer el mapa: {args.mapa}")
    px_per_cm = track.shape[1] / args.ancho_cm
    scale = MAP_VIEW_H / track.shape[0]

    sim = Simulation(track, px_per_cm, args.driver, auto_start(track, px_per_cm))
    create_trackbars(sim.brain.driver)
    cv2.namedWindow(WIN_MAP)

    clicks = {"pos": None, "new_start": None}

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        p = np.array([x / scale, y / scale])
        if clicks["pos"] is None:
            clicks["pos"] = p
        else:
            d = p - clicks["pos"]
            clicks["new_start"] = (clicks["pos"], math.atan2(-d[1], d[0]))
            clicks["pos"] = None

    cv2.setMouseCallback(WIN_MAP, on_mouse)
    speed = 1.0
    print(f"[SIM] Mapa {track.shape[1]}x{track.shape[0]} px = {args.ancho_cm:g} cm de ancho "
          f"({px_per_cm:.1f} px/cm) | Driver: {args.driver}")
    print("[SIM] q = salir, espacio = pausa, r = reiniciar, +/- = velocidad. "
          "Clic en el mapa: posición y luego dirección.")

    while True:
        if clicks["new_start"] is not None:
            sim.reset(clicks["new_start"])
            clicks["new_start"] = None

        threshold, roi_ratio, near_weight = read_trackbars(sim.brain)
        frame, line = sim.advance_one_frame(threshold, roi_ratio, near_weight)

        cv2.imshow(WIN_CAM, draw_overlay(frame, line, roi_ratio, sim.brain.snapshot(),
                                         sim.last_msg, config.SIM_CAM_FPS, config.DISPLAY_SCALE))
        cv2.imshow(WIN_MAP, draw_map(track, sim, scale, speed, clicks["pos"]))

        key = cv2.waitKey(max(1, int(1000 / (config.SIM_CAM_FPS * speed)))) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            sim.brain.set_paused(not sim.brain.snapshot()["paused"])
        if key == ord("r"):
            sim.reset()
        if key in (ord("+"), ord("=")):
            speed = min(speed * 2, 16)
        if key == ord("-"):
            speed = max(speed / 2, 0.25)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
