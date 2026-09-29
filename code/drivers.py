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

import config  # Parámetros del robot: tiempos de pulso, velocidades, formato de mensajes


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

    name = "pulsos"  # Identificador del driver (usado en argparse y config)

    def __init__(self):                                     # Inicializa el driver de pulsos
        self.max_turn_pulses = config.MAX_TURN_PULSES       # Máx. pulsos de giro por cada avance cuando |u|=1
        self.deadband = config.DEADBAND                     # Zona muerta: |u| menor que esto → recto
        self.cmd = config.PULSE_COMMANDS                    # Diccionario de letras: {"forward":"w", "left":"a", ...}
        self._acc = 0.0                                     # Acumulador sigma-delta para la parte fraccionaria de giros
        self._side = None                                   # Último lado de giro ('left' o 'right')
        self._queue = []                                    # Cola de acciones pendientes en el ciclo actual
        self._stopped = False                               # ¿Ya se envió el comando de parada?

    def stop_message(self):          # Devuelve el mensaje para detener el robot
        return self.cmd["stop"]      # Letra 'x' (por defecto)

    def reset(self):                 # Reinicia el estado interno del driver
        self._acc = 0.0              # Borra el acumulador sigma-delta
        self._side = None            # Olvida el último lado de giro
        self._queue.clear()          # Vacía la cola de acciones

    def next(self, u, moving):                                        # Genera el siguiente comando a enviar
        if not moving:                                                # Si el robot no debe moverse
            self.reset()                                              # Reinicia el estado del driver
            if self._stopped:                                         # Si ya envió 'x' antes
                return None, 0.02, (0.0, 0.0, 0.02)                  # No envía nada, espera 20ms
            self._stopped = True                                      # Marca que ya mandó la parada
            return self.cmd["stop"], 0.02, (0.0, 0.0, 0.02)          # Envía 'x', espera 20ms, movimiento nulo
        self._stopped = False                                         # Si está en movimiento, resetea el flag de parada

        if not self._queue:                                           # Si la cola de acciones está vacía
            self._queue = self._plan_cycle(u)                         # Planifica un nuevo ciclo de acciones según u
        action = self._queue.pop(0)                                   # Saca la primera acción de la cola

        if action == "forward":                                       # Si la acción es avanzar
            dur = config.PULSE_FORWARD_S                              # Duración del pulso de avance (ej: 100ms)
            motion = (config.SIM_FORWARD_CM_PER_S, 0.0, dur)         # Movimiento: velocidad lineal, sin giro
        else:                                                         # Si la acción es girar ('left' o 'right')
            dur = config.PULSE_TURN_S                                 # Duración del pulso de giro (ej: 30ms)
            sign = 1.0 if action == "left" else -1.0                  # Izquierda = ω positivo, derecha = ω negativo
            motion = (0.0, sign * config.SIM_TURN_DEG_PER_S, dur)    # Movimiento: sin avance, solo giro
        return self.cmd[action], dur + config.LINK_MARGIN_S, motion   # Devuelve letra, tiempo de espera (pulso + margen BT), y movimiento

    def _plan_cycle(self, u):                              # Planifica las acciones de un ciclo completo
        mag = abs(u)                                       # Magnitud del mando de giro (0 a 1)
        if mag < self.deadband:                            # Si está dentro de la zona muerta
            self._acc = 0.0                                # Resetea el acumulador
            return ["forward"]                             # Solo avanza recto

        side = "right" if u > 0 else "left"                # Determina el lado de giro según el signo de u
        if side != self._side:                             # Si cambió de lado respecto al ciclo anterior
            self._acc = 0.0                                # Resetea el acumulador (no arrastrar giros del otro lado)
            self._side = side                              # Guarda el nuevo lado

        self._acc += mag * self.max_turn_pulses            # Suma al acumulador: fracción de giros deseados
        k = int(self._acc)                                 # Parte entera = número de pulsos de giro este ciclo
        self._acc -= k                                     # Guarda la parte fraccionaria para el próximo ciclo

        if mag >= config.SPIN_THRESHOLD:                   # Si |u| es muy grande (curva cerrada, ej: >0.85)
            return [side] * max(k, 1)                      # Solo gira sin avanzar (mínimo 1 pulso)
        return [side] * k + ["forward"]                    # k pulsos de giro + 1 avance al final


class SpeedDriver:
    """
    Firmware que acepte velocidad por rueda (control diferencial clásico):

        izquierda = base + u·base        derecha = base − u·base

    saturadas a [0, MAX_SPEED]. Con u = +1 la izquierda va al doble y la
    derecha se detiene: giro máximo a la derecha sin retroceder.
    """

    name = "velocidades"  # Identificador del driver

    def __init__(self):                           # Inicializa el driver de velocidades
        self.base = config.BASE_SPEED             # Velocidad base de ambas ruedas (0-255)
        self.deadband = config.DEADBAND           # Zona muerta: |u| menor que esto → ambas ruedas iguales
        self._stopped = False                     # ¿Ya se envió el comando de parada?

    def stop_message(self):                                   # Devuelve el mensaje para detener el robot
        return config.SPEED_FORMAT.format(left=0, right=0)    # "V:0,0\n"

    def reset(self):          # No tiene estado interno que reiniciar
        pass                  # (a diferencia de PulseDriver que tiene acumulador)

    def speeds(self, u):                                              # Calcula la velocidad de cada rueda dada u
        if abs(u) < self.deadband:                                    # Si u está en la zona muerta
            u = 0.0                                                   # Trata como recto (sin giro)
        left = self.base + u * self.base                              # Rueda izquierda: más rápida si u>0 (gira a la derecha)
        right = self.base - u * self.base                             # Rueda derecha: más lenta si u>0 (gira a la derecha)
        clip = lambda s: int(max(0, min(config.MAX_SPEED, s)))        # Función para saturar entre 0 y 255
        return clip(left), clip(right)                                # Devuelve ambas velocidades saturadas

    def next(self, u, moving):                                                    # Genera el siguiente comando a enviar
        period = config.SPEED_PERIOD_S                                            # Periodo entre mensajes (50ms = 20 Hz)
        if not moving:                                                            # Si el robot no debe moverse
            if self._stopped:                                                     # Si ya mandó parada antes
                return None, period, (0.0, 0.0, period)                           # No envía nada, espera un periodo
            self._stopped = True                                                  # Marca que ya mandó la parada
            return config.SPEED_FORMAT.format(left=0, right=0), period, (0.0, 0.0, period)  # Envía "V:0,0\n"
        self._stopped = False                                                     # Si está en movimiento, resetea flag de parada

        left, right = self.speeds(u)                                              # Calcula velocidades de cada rueda
        # Cinemática diferencial: v = (vl + vr)/2, ω = (vr − vl)/L
        k = config.SIM_FORWARD_CM_PER_S / max(config.BASE_SPEED, 1)              # Factor de conversión: PWM → cm/s
        vl, vr = left * k, right * k                                             # Velocidad real de cada rueda en cm/s
        w_deg = (vr - vl) / config.SIM_WHEEL_BASE_CM * 57.2958                   # Velocidad angular en °/s (rad→° = ×180/π ≈ 57.3)
        motion = ((vl + vr) / 2.0, w_deg, period)                                # Movimiento: (vel_lineal, vel_angular, duración)
        return config.SPEED_FORMAT.format(left=left, right=right), period, motion  # Devuelve mensaje, espera y movimiento


DRIVERS = {PulseDriver.name: PulseDriver, SpeedDriver.name: SpeedDriver}  # Registro de drivers disponibles por nombre


def make_driver(name):                                                              # Fábrica: crea un driver dado su nombre
    try:                                                                            # Intenta buscar el driver en el registro
        return DRIVERS[name]()                                                      # Lo instancia y devuelve
    except KeyError:                                                                # Si el nombre no existe
        raise ValueError(f"Driver desconocido {name!r}; opciones: {list(DRIVERS)}")  # Error descriptivo
