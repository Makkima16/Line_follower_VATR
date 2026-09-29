"""
Traducción del mando u ∈ [-1, 1] al "idioma" del robot.

El control (control.py) no sabe qué robot hay del otro lado. Solo produce
u = cuánto y hacia dónde girar. Cada driver convierte u en los mensajes que
entiende un firmware concreto. Si el robot entregado habla otro protocolo,
basta con escribir otro driver con el mismo método ``next()``.

``next(u, moving)`` devuelve ``(mensaje | None, espera_s, movimiento)``:
  * mensaje: texto a enviar (None = no enviar nada esta vez).
  * espera_s: cuánto esperar antes de pedir el siguiente.
  * movimiento: ``(v_cm_s, w_deg_s, dur_s)`` aproximado, para el simulador.
"""

import config


class PulseDriver:
    """
    Firmware del curso (arduinoFinal.ino): solo letras con pulsos fijos.

      'w' → adelante PULSE_FORWARD_S     'a'/'d' → gira en sitio PULSE_TURN_S

    Como no hay velocidad por rueda, el giro proporcional se logra en el
    tiempo: por cada 'w' se intercalan en promedio |u|·MAX_TURN_PULSES
    pulsos de giro. La parte fraccionaria se acumula (sigma-delta): con
    |u|·MAX = 0.5 sale un giro cada dos ciclos; con 1.5 alternan 1 y 2.
    Es un PWM a nivel de comandos, así que no hay zig-zag binario.
    """

    name = "pulsos"

    def __init__(self):
        self.max_turn_pulses = config.MAX_TURN_PULSES
        self.deadband = config.DEADBAND
        self.cmd = config.PULSE_COMMANDS
        self._acc = 0.0
        self._side = None
        self._queue = []
        self._stopped = False

    def stop_message(self):
        return self.cmd["stop"]

    def reset(self):
        self._acc = 0.0
        self._side = None
        self._queue.clear()

    def next(self, u, moving):
        if not moving:
            self.reset()
            if self._stopped:
                return None, 0.02, (0.0, 0.0, 0.02)
            self._stopped = True
            return self.cmd["stop"], 0.02, (0.0, 0.0, 0.02)
        self._stopped = False

        if not self._queue:
            self._queue = self._plan_cycle(u)
        action = self._queue.pop(0)

        if action == "forward":
            dur = config.PULSE_FORWARD_S
            motion = (config.SIM_FORWARD_CM_PER_S, 0.0, dur)
        else:
            dur = config.PULSE_TURN_S
            sign = 1.0 if action == "left" else -1.0    # Antihorario = positivo
            motion = (0.0, sign * config.SIM_TURN_DEG_PER_S, dur)
        return self.cmd[action], dur + config.LINK_MARGIN_S, motion

    def _plan_cycle(self, u):
        mag = abs(u)
        if mag < self.deadband:
            self._acc = 0.0
            return ["forward"]

        side = "right" if u > 0 else "left"
        if side != self._side:
            self._acc = 0.0          # Cambió de lado: no arrastrar el resto
            self._side = side

        self._acc += mag * self.max_turn_pulses
        k = int(self._acc)
        self._acc -= k

        if mag >= config.SPIN_THRESHOLD:
            return [side] * max(k, 1)          # Curva cerrada: girar sin avanzar
        return [side] * k + ["forward"]


class SpeedDriver:
    """
    Firmware que acepte velocidad por rueda (control diferencial clásico):

        izquierda = base + u·base        derecha = base − u·base

    saturadas a [0, MAX_SPEED]. Con u = +1 la izquierda va al doble y la
    derecha se detiene: giro máximo a la derecha sin retroceder.
    """

    name = "velocidades"

    def __init__(self):
        self.base = config.BASE_SPEED
        self.deadband = config.DEADBAND
        self._stopped = False

    def stop_message(self):
        return config.SPEED_FORMAT.format(left=0, right=0)

    def reset(self):
        pass

    def speeds(self, u):
        if abs(u) < self.deadband:
            u = 0.0
        left = self.base + u * self.base
        right = self.base - u * self.base
        clip = lambda s: int(max(0, min(config.MAX_SPEED, s)))
        return clip(left), clip(right)

    def next(self, u, moving):
        period = config.SPEED_PERIOD_S
        if not moving:
            if self._stopped:
                return None, period, (0.0, 0.0, period)
            self._stopped = True
            return config.SPEED_FORMAT.format(left=0, right=0), period, (0.0, 0.0, period)
        self._stopped = False

        left, right = self.speeds(u)
        # Cinemática diferencial: v = (vl + vr)/2, ω = (vr − vl)/L
        k = config.SIM_FORWARD_CM_PER_S / max(config.BASE_SPEED, 1)
        vl, vr = left * k, right * k
        w_deg = (vr - vl) / config.SIM_WHEEL_BASE_CM * 57.2958
        motion = ((vl + vr) / 2.0, w_deg, period)
        return config.SPEED_FORMAT.format(left=left, right=right), period, motion


DRIVERS = {PulseDriver.name: PulseDriver, SpeedDriver.name: SpeedDriver}


def make_driver(name):
    try:
        return DRIVERS[name]()
    except KeyError:
        raise ValueError(f"Driver desconocido {name!r}; opciones: {list(DRIVERS)}")
