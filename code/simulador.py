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
  python simulador.py ../pistas/final.jpeg --ancho-cm 155 --inicio 807 1300 87

Teclas: q = salir, espacio = pausa, r = reiniciar, + / - = más/menos rápido.
Ratón (ventana "Mapa"): clic = posición inicial, segundo clic = hacia dónde mira.
"""

import argparse
import math
from collections import deque

import cv2
import numpy as np

import config
import mapa
import senales
import vision
from control import Brain, State
from drivers import DRIVERS, make_driver
from main import WIN_TUNE, create_trackbars, draw_overlay, read_trackbars

WIN_MAP = "Mapa"
WIN_CAM = "Camara virtual"
WIN_ODOM = "Mapa (odometria)"
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
    """Cámara falsa: "fotografía" el trozo de mapa que vería el celular del robot."""

    def __init__(self, track, px_per_cm):
        self.track = track
        self.px_per_cm = px_per_cm
        self.w, self.h = config.PROCESS_WIDTH, config.SIM_CAM_HEIGHT_PX
        # Esquinas destino = esquinas del frame (CRÍTICO: float32, getPerspectiveTransform no acepta otro tipo)
        self.dst = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]])
        # Fuera del mapa se ve "papel": el color mediano de la foto
        self.paper = tuple(int(c) for c in np.median(track.reshape(-1, 3), axis=0))

    def capture(self, pos, theta):
        """Devuelve el frame que vería la cámara con el robot en pos mirando a theta."""
        src = camera_footprint(pos, theta, self.px_per_cm)
        # Homografía 3x3 que lleva el trapecio del piso al rectángulo del frame
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
    # Sin contornos: pose por defecto (abajo al centro, mirando arriba) en vez de fallar en max()
    if not cnts:
        return np.array([w / 2, h * 0.9]), math.pi / 2

    def touches_side(c):
        x, _, cw, _ = cv2.boundingRect(c)
        return x <= 2 or x + cw >= w - 2

    # "or cnts": si todos tocan el borde se usan todos (evita max() sobre una lista vacía)
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
    """
    Mundo simulado: pose del robot, reloj propio, Brain y cámara virtual.
    Reproduce los dos hilos de main.py (video y envío) pero en un solo bucle.
    """

    def __init__(self, track, px_per_cm, driver_name, start):
        self.cam = VirtualCamera(track, px_per_cm)
        self.px_per_cm = px_per_cm
        self.driver_name = driver_name
        self.start = start
        self.reset()

    def reset(self, start=None):
        """Vuelve a la pose inicial (o a una nueva) y crea un Brain limpio."""
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
        self.last_signal = None

    def _integrate(self, h):
        """Avanza la física h segundos con el modelo uniciclo (x += v·cosθ·h, θ += ω·h)."""
        # El comando solo actúa durante su duración; después el robot queda quieto.
        # Se integra solo la parte del paso que cae dentro del comando (si no,
        # un pulso de 30 ms con pasos de 5 ms podía durar un paso de más).
        v, w_deg = self.motion
        active = min(h, max(0.0, self.motion_until - self.t))
        if active > 0 and (v or w_deg):
            fwd, _ = axes(self.theta)
            self.pos += fwd * v * self.px_per_cm * active
            self.theta += math.radians(w_deg) * active
            self.distance_cm += abs(v) * active
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
                _, line, label, partial = self.pending.popleft()
                self.brain.on_signal(label, self.t, partial)
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
                signal = senales.detect_signal(frame)
                self.pending.append((self.t + config.SIM_LATENCY_S, line,
                                     signal.label if signal is not None else None,
                                     signal is not None and signal.partial))
                if np.linalg.norm(self.pos - self.trail[-1]) > 2:
                    self.trail.append(self.pos.copy())
                self.last_signal = signal
                return frame, line


# ─────────────────────────────────────────────────────────────────────────────
#  DIBUJO
# ─────────────────────────────────────────────────────────────────────────────
# CRÍTICO: debe tener un color para cada estado que pueda aparecer en draw_map,
# si falta uno (ej. State.PARE) la búsqueda STATE_COLORS[...] da KeyError.
STATE_COLORS = {State.SIGUIENDO: (0, 200, 0), State.BUSCANDO: (0, 140, 255),
                State.DETENIDO: (0, 0, 255), State.PARE: (0, 0, 220),
                State.MEDIA_VUELTA: (255, 0, 255), State.FIN: (200, 200, 200)}


def draw_map(track, sim, scale, speed, clicked):
    """Dibuja el mapa con el rastro, el campo de visión de la cámara, el robot y los datos."""
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
    """Lee la imagen del mapa, su ancho real en cm y el driver a usar."""
    p = argparse.ArgumentParser(description="Simulador de lazo cerrado del seguidor")
    p.add_argument("mapa", help="Foto cenital de la pista (línea oscura sobre fondo claro)")
    p.add_argument("--ancho-cm", type=float, default=config.SIM_MAP_WIDTH_CM,
                   help="Ancho real que representa la imagen (escala del mapa)")
    p.add_argument("--driver", choices=list(DRIVERS), default=config.DRIVER)
    p.add_argument("--inicio", nargs=3, type=float, metavar=("X", "Y", "ANG"),
                   help="Pose inicial: X Y en píxeles de la foto y ANG en grados "
                        "(0 = derecha, 90 = arriba, 180 = izquierda, 270 = abajo). "
                        "Sin esto se usa auto_start (solo sirve si la línea tiene extremos).")
    return p.parse_args()


def main():
    """Carga el mapa, crea la simulación y corre el bucle de ventanas/teclado."""
    args = parse_args()
    track = cv2.imread(args.mapa)
    # CRÍTICO: imread no lanza error si el archivo no existe, devuelve None;
    # sin esta comprobación fallaría más adelante con un error confuso.
    if track is None:
        raise SystemExit(f"No se pudo leer el mapa: {args.mapa}")
    # Escala del mapa: píxeles de la foto por cada cm real
    px_per_cm = track.shape[1] / args.ancho_cm
    scale = MAP_VIEW_H / track.shape[0]

    if args.inicio is not None:
        x, y, ang = args.inicio
        start = (np.array([x, y]), math.radians(ang))
    else:
        start = auto_start(track, px_per_cm)
    sim = Simulation(track, px_per_cm, args.driver, start)
    create_trackbars(sim.brain.driver)
    cv2.namedWindow(WIN_MAP)

    clicks = {"pos": None, "new_start": None}

    def on_mouse(event, x, y, _flags, _param):
        """Primer clic = posición inicial; segundo clic = hacia dónde mira el robot."""
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
                                         sim.last_msg, config.SIM_CAM_FPS, config.DISPLAY_SCALE,
                                         sim.last_signal))
        cv2.imshow(WIN_MAP, draw_map(track, sim, scale, speed, clicks["pos"]))
        cv2.imshow(WIN_ODOM, mapa.draw_map(sim.brain.map_snapshot()))

        # CRÍTICO: max(1, ...) porque waitKey(0) espera una tecla para siempre
        # y congelaría la simulación.
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
