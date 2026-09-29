"""
Cerebro del seguidor: control PD y comportamiento ante línea perdida.

Produce un único mando u ∈ [-1, 1] (+ = girar a la derecha) que después
un driver (drivers.py) traduce al protocolo del robot.
"""

import enum
import threading
import time

import config


class PDController:
    """
    u = Kp·e + Kd·de/dt, saturado a [-1, 1].

    La derivada frena el giro cuando el error ya se está corrigiendo (evita
    pasarse de la línea). Se filtra con una media exponencial porque la
    cámara mete ruido de un frame a otro.
    """

    def __init__(self, kp=config.KP, kd=config.KD):
        self.kp = kp
        self.kd = kd
        self.reset()

    def reset(self):
        self._prev_error = None
        self._prev_t = None
        self._d_filtered = 0.0

    def update(self, error, now):
        if self._prev_error is None:
            d = 0.0
        else:
            d = (error - self._prev_error) / max(now - self._prev_t, 1e-3)
        self._d_filtered = 0.5 * self._d_filtered + 0.5 * d
        self._prev_error, self._prev_t = error, now
        u = self.kp * error + self.kd * self._d_filtered
        return max(-1.0, min(1.0, u))


class State(enum.Enum):
    SIGUIENDO = "SIGUIENDO"      # Línea a la vista, control PD
    BUSCANDO = "BUSCANDO"        # Línea perdida: gira hacia el último lado visto
    DETENIDO = "DETENIDO"        # Sin línea demasiado tiempo: parado por seguridad


class Brain:
    """
    Estado compartido entre el hilo de video y el de envío.

    * Hilo de video → ``on_frame()`` con cada detección nueva.
    * Hilo de envío → ``next_command()`` cada vez que toca mandar algo.

    Los tiempos se comparan con ``time.monotonic()``: nada bloquea el video.
    """

    def __init__(self, driver):
        self.lock = threading.Lock()
        self.pd = PDController()
        self.driver = driver
        self.state = State.SIGUIENDO
        self.u = 0.0
        self.error = 0.0
        self.line_found = False
        self.last_side = 1               # +1 derecha, -1 izquierda
        self.lost_since = None
        self.paused = False

    # ── Hilo de video ──────────────────────────────────────────────────────
    def on_frame(self, line, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if line is not None:
                self._on_line(line, now)
            else:
                self._on_lost(now)

    def _on_line(self, line, now):
        if self.state != State.SIGUIENDO:
            self.pd.reset()
            self.driver.reset()
            print("[CONTROL] Línea encontrada")
        self.state = State.SIGUIENDO
        self.line_found = True
        self.lost_since = None
        self.error = line.error
        self.u = self.pd.update(line.error, now)
        if abs(self.u) >= config.DEADBAND:
            self.last_side = 1 if self.u > 0 else -1

    def _on_lost(self, now):
        self.line_found = False
        if self.lost_since is None:
            self.lost_since = now
        lost_for = now - self.lost_since

        if lost_for < config.LOST_GRACE_S:
            return                                  # Parpadeo: mantener la última u
        if lost_for < config.LOST_STOP_S:
            if self.state != State.BUSCANDO:
                self.driver.reset()
                side = "derecha" if self.last_side > 0 else "izquierda"
                print(f"[CONTROL] Línea perdida, buscando a la {side}")
            self.state = State.BUSCANDO
            self.u = config.RECOVERY_U * self.last_side
        elif self.state != State.DETENIDO:
            self.state = State.DETENIDO
            print("[CONTROL] Sin línea demasiado tiempo → detenido")

    # ── Hilo de envío ──────────────────────────────────────────────────────
    def next_command(self):
        with self.lock:
            moving = self.state != State.DETENIDO and not self.paused
            return self.driver.next(self.u, moving)

    # ── Utilidades ─────────────────────────────────────────────────────────
    def set_paused(self, paused):
        with self.lock:
            self.paused = paused

    def snapshot(self):
        with self.lock:
            return {"state": self.state, "u": self.u, "error": self.error,
                    "line_found": self.line_found, "paused": self.paused}
