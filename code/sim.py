"""
Vista "Movimiento": cómo se movería el robot con los comandos enviados.

Integra la cinemática de un robot diferencial (modelo uniciclo):
    x += v·cos(θ)·dt     y += v·sin(θ)·dt     θ += ω·dt
con el (v, ω, dt) que reporta el driver en cada comando. Es una
aproximación (depende de SIM_* en config.py), pensada para ver sin robot si
las decisiones tienen sentido: recto cuando la línea está centrada, curva
suave cuando se desvía, giro hacia el último lado cuando se pierde.
"""

import math
import threading
from collections import deque

import cv2
import numpy as np

SIZE = 360            # Lado de la ventana (px)
PX_PER_CM = 3.0       # Escala del dibujo
TRAIL_LEN = 1500


class MotionPreview:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.x = self.y = 0.0
            self.theta = math.pi / 2          # Mirando "hacia arriba"
            self.trail = deque([(0.0, 0.0)], maxlen=TRAIL_LEN)
            self.messages = deque(maxlen=28)
            self.v = self.w = 0.0

    def add(self, motion, message):
        """Llamado desde el hilo de envío con cada comando."""
        v, w_deg, dt = motion
        with self._lock:
            # Integración en sub-pasos para que los giros largos queden suaves
            steps = max(1, int(dt / 0.01))
            h = dt / steps
            for _ in range(steps):
                self.x += v * math.cos(self.theta) * h
                self.y += v * math.sin(self.theta) * h
                self.theta += math.radians(w_deg) * h
            self.v, self.w = v, w_deg
            if v != 0 or w_deg != 0:
                self.trail.append((self.x, self.y))
            if message is not None:
                self.messages.append(message.strip() or "?")

    def draw(self):
        img = np.full((SIZE, SIZE + 120, 3), 30, np.uint8)
        with self._lock:
            cx, cy = self.x, self.y
            trail = list(self.trail)
            theta, v, w = self.theta, self.v, self.w
            msgs = list(self.messages)

        # La vista sigue al robot: (cx, cy) queda en el centro
        def to_px(px, py):
            return (int(SIZE / 2 + (px - cx) * PX_PER_CM),
                    int(SIZE / 2 - (py - cy) * PX_PER_CM))

        # Cuadrícula de 10 cm
        step = 10 * PX_PER_CM
        off_x = (-cx * PX_PER_CM) % step
        off_y = (cy * PX_PER_CM) % step
        for i in range(int(SIZE / step) + 2):
            gx, gy = int(off_x + i * step), int(off_y + i * step)
            cv2.line(img, (gx, 0), (gx, SIZE), (50, 50, 50), 1)
            cv2.line(img, (0, gy), (SIZE, gy), (50, 50, 50), 1)

        if len(trail) > 1:
            pts = np.array([to_px(*p) for p in trail], np.int32)
            cv2.polylines(img, [pts], False, (0, 200, 255), 2, cv2.LINE_AA)

        # Robot como triángulo orientado
        c = np.array(to_px(cx, cy), float)
        fwd = np.array([math.cos(theta), -math.sin(theta)])
        side = np.array([-fwd[1], fwd[0]])
        tri = np.array([c + fwd * 16, c - fwd * 10 + side * 10, c - fwd * 10 - side * 10], np.int32)
        cv2.fillPoly(img, [tri], (0, 255, 0))

        cv2.rectangle(img, (0, 0), (SIZE, 44), (0, 0, 0), -1)
        cv2.putText(img, f"v={v:5.1f} cm/s  w={w:+6.1f} deg/s", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "r = reiniciar trayectoria", (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

        # Historial de mensajes enviados (columna derecha)
        cv2.putText(img, "Enviado:", (SIZE + 8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)
        for i, m in enumerate(reversed(msgs[-26:])):
            cv2.putText(img, m[:12], (SIZE + 8, 38 + i * 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 200, 255) if i == 0 else (140, 140, 140), 1, cv2.LINE_AA)
        return img
