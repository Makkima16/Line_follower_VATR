"""
Cerebro del seguidor: control PD, señales, línea perdida y ramificaciones.

Produce un único mando u ∈ [-1, 1] (+ = girar a la derecha) que después
un driver (drivers.py) traduce al protocolo del robot.
"""

import enum
import threading
import time

import config
from mapa import Navigator


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
        # CRÍTICO: reset() crea _prev_error, _prev_t y _d_filtered; sin esta
        # llamada update() fallaría con AttributeError.
        self.reset()

    def reset(self):
        """Olvida la historia (error y tiempo anteriores) para que la derivada arranque en 0."""
        self._prev_error = None
        self._prev_t = None
        self._d_filtered = 0.0

    def update(self, error, now):
        """
        Calcula el mando u para el error actual.

        error : desviación de la línea en [-1, 1] (> 0 = línea a la derecha).
        now   : tiempo en segundos (time.monotonic()), para calcular dt.
        """
        if self._prev_error is None:
            d = 0.0                      # Primer dato: no hay derivada todavía
        else:
            # de/dt = (e - e_anterior) / dt
            # CRÍTICO: max(dt, 1e-3) evita dividir entre 0 si llegan dos frames
            # con el mismo tiempo.
            d = (error - self._prev_error) / max(now - self._prev_t, 1e-3)
        # Filtro exponencial: 50 % valor anterior + 50 % valor nuevo
        a = config.D_SMOOTHING
        self._d_filtered = a * self._d_filtered + (1.0 - a) * d
        self._prev_error, self._prev_t = error, now
        u = self.kp * error + self.kd * self._d_filtered
        return max(-1.0, min(1.0, u))    # Saturación a [-1, 1]


class State(enum.Enum):
    """Estados posibles del robot (máquina de estados del cerebro)."""
    SIGUIENDO = "SIGUIENDO"      # Línea a la vista, control PD
    BUSCANDO = "BUSCANDO"        # Línea perdida: gira hacia el último lado visto
    DETENIDO = "DETENIDO"        # Sin línea demasiado tiempo: parado por seguridad
    PARE = "PARE"                # Señal de Pare: detenido stop_duration s y reanuda solo
    MEDIA_VUELTA = "MEDIA VUELTA"  # Callejón sin salida: gira en sitio para regresar
    FIN = "FIN"                  # Mapa agotado: no queda ninguna rama por explorar


class Brain:
    """
    Estado compartido entre el hilo de video y el de envío.

    * Hilo de video → ``on_signal()`` y ``on_frame()`` con cada detección nueva.
    * Hilo de envío → ``next_command()`` cada vez que toca mandar algo; de paso
      integra la odometría del mapa con el movimiento de ese comando.

    Los tiempos se comparan con ``time.monotonic()``: nada bloquea el video.
    """

    def __init__(self, driver):
        # CRÍTICO: el candado protege los datos que leen/escriben los dos hilos a
        # la vez. Sin él el hilo de envío podría leer un estado a medio actualizar.
        self.lock = threading.Lock()
        self.pd = PDController()
        self.driver = driver             # Traduce u al mensaje del robot (drivers.py)
        self.nav = Navigator()           # Odometría + mapa de cruces (mapa.py)
        self.state = State.SIGUIENDO
        self.u = 0.0                     # Mando actual en [-1, 1]
        self.error = 0.0                 # Último error medido
        self.line_found = False
        self.last_side = 1               # +1 derecha, -1 izquierda
        self.lost_since = None           # Momento en que se perdió la línea
        self.paused = False
        self._error_f = 0.0
        self._error_init = False
        # Ramificaciones y callejones
        self._junction_frames = 0        # Frames seguidos viendo 2+ salidas
        self._dead_frames = 0            # Frames seguidos viendo la línea terminar
        self._line_ending = False        # En el último frame la línea terminaba a la vista
        self._uturn_from = 0.0           # Giro acumulado de la odometría al empezar la media vuelta
        self._no_junction_until = 0.0    # Distancia (odometría) hasta la que se ignoran cruces
        # Señales de tráfico (objetivos 5 y 6 del reto)
        self.signal = None               # "PARE" o "SIGA", última señal que actuó
        self.stop_duration = config.STOP_DURATION_S   # [TB]
        self.stop_until = 0.0            # Hasta cuándo permanecer en PARE
        self.signal_cover_until = 0.0    # Hasta cuándo la línea puede estar tapada por la señal
        self._sig_label = None           # Etiqueta vista en los últimos frames
        self._sig_count = 0              # Frames seguidos con esa etiqueta
        self._sig_last_seen = -1e9       # Última vez que se vio alguna señal
        self._sig_armed = {"PARE": True, "SIGA": True}
        self._now = 0.0                  # Reloj del último frame (el simulador usa uno propio)

    # ── Hilo de video ──────────────────────────────────────────────────────
    def on_frame(self, line, now=None):
        """
        Recibe la detección de un frame (LineResult o None) y actualiza el estado.
        ``now`` se puede pasar a mano (el simulador usa un reloj propio).
        """
        now = time.monotonic() if now is None else now
        with self.lock:
            self._now = now
            if self.state == State.FIN:
                self.u = 0.0
                return
            if self.state == State.PARE:
                if now < self.stop_until:
                    self.u = 0.0           # Quieto; next_command tampoco lo mueve
                    return
                print("[CONTROL] Fin del PARE → reanudando")
                self._resume()
            if self.state == State.MEDIA_VUELTA:
                self._on_uturn(line, now)
            elif line is not None:
                self._on_line(line, now)
            else:
                self._on_lost(now)

    def _resume(self):
        """Vuelve a SIGUIENDO sin heredar la derivada ni el acumulador del driver."""
        self.state = State.SIGUIENDO
        self.lost_since = None
        self.pd.reset()
        self.driver.reset()
        self._error_init = False

    def _on_line(self, line, now):
        """
        Línea visible: revisa si hay ramificación o callejón y calcula u con el PD.

        El error normal es el de detect_line; mientras hay una rama elegida en
        un cruce, lo reemplaza nav.steer_error() (apuntar a esa rama).
        """
        if self.state != State.SIGUIENDO:
            print("[CONTROL] Línea encontrada")
            self._resume()
        self.line_found = True
        self.lost_since = None
        covered = now < self.signal_cover_until       # La señal puede cortar la línea
        # Callejón: la mancha no sale por ningún borde, termina en la mitad cercana
        # y no se ve otro trozo de línea más allá (si se ve, es un hueco en la cinta)
        self._line_ending = (not line.exits and line.reach < config.DEAD_END_REACH
                             and line.others == 0 and not covered)
        free = self.nav.commit is None                 # No hay una rama fijada

        # Barra transversal vista entera: desde la cámara se ve que sus brazos
        # terminan, no son camino. Al acercarse se salen por los costados y
        # parecerían ramas, así que se recuerda hasta haberla pasado.
        if line.crossbar and len(line.exits) <= 1:
            self._no_junction_until = self.nav.odom.dist + config.CROSSBAR_IGNORE_CM
        can_branch = free and self.nav.odom.dist >= self._no_junction_until

        # Ramificación: 2+ salidas durante JUNCTION_CONFIRM_FRAMES seguidos
        self._junction_frames = (self._junction_frames + 1
                                 if can_branch and len(line.exits) >= 2 else 0)
        if self._junction_frames >= config.JUNCTION_CONFIRM_FRAMES:
            self._junction_frames = 0
            self.nav.on_junction(line)
            self.pd.reset()

        # Callejón: DEAD_END_FRAMES seguidos viendo terminar la línea
        self._dead_frames = self._dead_frames + 1 if free and self._line_ending else 0
        if self._dead_frames >= config.DEAD_END_FRAMES:
            self._dead_frames = 0
            self._start_uturn()
            return

        error = self.nav.steer_error(line)
        raw = line.error if error is None else error
        a = config.ERROR_SMOOTHING
        if not self._error_init:
            self._error_f = raw
            self._error_init = True
        else:
            self._error_f = a * self._error_f + (1.0 - a) * raw
        self.error = self._error_f
        self.u = self.pd.update(self.error, now)
        # Recordar hacia dónde se giraba (para buscar por ese lado si se pierde)
        if abs(self.u) >= config.DEADBAND:
            self.last_side = 1 if self.u > 0 else -1

    def _on_lost(self, now):
        """
        Línea no visible. Según cuánto tiempo lleva perdida:
          < LOST_GRACE_S : parpadeo, se mantiene la última u.
          si la línea venía terminando: callejón → media vuelta.
          < LOST_STOP_S  : BUSCANDO, gira suave hacia el último lado visto.
          después        : DETENIDO por seguridad.
        """
        self.line_found = False
        self._error_init = False
        self._junction_frames = self._dead_frames = 0

        # Si la línea desaparece justo después de ver una señal,
        # asumimos que está tapada por la señal que estamos cruzando.
        # Seguimos recto (u=0) en vez de girar, durante la gracia configurada.
        if self.state == State.SIGUIENDO and now < self.signal_cover_until:
            self.u = 0.0
            self.lost_since = None
            return

        # CRÍTICO: se guarda el instante de la pérdida antes de restar; si
        # lost_since siguiera en None, la resta de abajo daría TypeError.
        if self.lost_since is None:
            self.lost_since = now
        lost_for = now - self.lost_since

        if lost_for < config.LOST_GRACE_S:
            return                                  # Parpadeo: mantener la última u
        if self._line_ending and self.nav.commit is None:
            self._line_ending = False               # Se pasó del final de la línea
            self._start_uturn()
            return
        if lost_for < config.LOST_STOP_S:
            if self.state != State.BUSCANDO:
                self.driver.reset()
                side = "derecha" if self.last_side > 0 else "izquierda"
                print(f"[CONTROL] Línea perdida, buscando a la {side}")
            self.state = State.BUSCANDO
            if self.nav.commit is not None:         # Girando en un cruce hacia la rama elegida
                self.u = self.nav.steer_error(None)
            else:
                self.u = config.RECOVERY_U * self.last_side   # Giro suave hacia el último lado
        elif self.state != State.DETENIDO:
            self.state = State.DETENIDO
            print("[CONTROL] Sin línea demasiado tiempo → detenido")

    def _start_uturn(self):
        """Callejón confirmado: se anota en el mapa y se gira en sitio para regresar."""
        if not self.nav.on_dead_end():
            self.state = State.FIN
            self.u = 0.0
            print("[CONTROL] Mapa explorado por completo → FIN")
            return
        self.state = State.MEDIA_VUELTA
        self._uturn_from = self.nav.odom.turned
        self.u = config.UTURN_U * self.last_side
        self.driver.reset()
        print("[CONTROL] Callejón sin salida → media vuelta")

    def _on_uturn(self, line, now):
        """
        Gira en sitio. La línea no se acepta hasta haber girado UTURN_MIN_DEG
        (al principio la cámara todavía ve el trozo del callejón); con la
        odometría se sabe cuánto se lleva girado: |Σ ω·dt|. Además la línea
        tiene que salir por algún borde de la ROI: si termina a la vista es
        otra vez el cabo del callejón, no el camino de regreso.
        """
        turned = abs(self.nav.odom.turned - self._uturn_from)
        if turned >= config.UTURN_MIN_DEG and line is not None and line.exits:
            print(f"[CONTROL] Media vuelta completa ({turned:.0f}°), regresando")
            self._resume()
            self._on_line(line, now)
        elif turned > config.UTURN_MAX_DEG:
            print("[CONTROL] Media vuelta sin encontrar la línea → buscando")
            self.state = State.BUSCANDO
            self.lost_since = now - config.LOST_GRACE_S

    # ── Hilo de envío ──────────────────────────────────────────────────────
    def next_command(self):
        """
        Pide al driver el siguiente comando y actualiza la odometría con el
        movimiento que ese comando va a producir.
        Devuelve (mensaje, segundos_de_espera, (v, ω, duración)).
        """
        with self.lock:
            # Solo se mueve si no está detenido, en PARE, en FIN ni en pausa
            moving = (self.state not in (State.DETENIDO, State.PARE, State.FIN)
                      and not self.paused)
            msg, wait, motion = self.driver.next(self.u, moving)
            self.nav.on_motion(*motion)
            return msg, wait, motion

    # ── Utilidades ─────────────────────────────────────────────────────────
    def set_paused(self, paused):
        """Pausa o reanuda el robot (tecla espacio)."""
        with self.lock:
            self.paused = paused

    def on_signal(self, label, now, partial=False):
        """
        Llamar en CADA frame con la etiqueta de senales.detect_signal() (o None).

        partial=True: la señal está cortada por el borde del frame. No se actúa
        (su forma no es fiable), pero se sabe que hay una señal delante: puede
        tapar la línea y, en esta pista, siempre va sobre una barra transversal,
        así que no se aceptan cruces hasta pasarla.

        * Se exige la misma etiqueta SIGNAL_CONFIRM_FRAMES frames seguidos.
        * Después de actuar, la señal queda "desarmada" hasta que deje de
          verse SIGNAL_COOLDOWN_S: al reanudar tras el PARE el robot sigue
          viendo la misma señal mientras pasa por encima y no debe volver a parar.
        """
        with self.lock:
            if label is not None and partial:
                self._sig_last_seen = now          # Sigue a la vista: no re-armar
                self.signal_cover_until = now + config.SIGNAL_COVER_GRACE_S
                self._no_junction_until = self.nav.odom.dist + config.CROSSBAR_IGNORE_CM
                self._sig_count = 0
                return
            if label is None:
                self._sig_count = 0
                if now - self._sig_last_seen > config.SIGNAL_COOLDOWN_S:
                    self._sig_armed = {"PARE": True, "SIGA": True}
                return
            self._sig_last_seen = now
            self.signal_cover_until = now + config.SIGNAL_COVER_GRACE_S
            self._no_junction_until = self.nav.odom.dist + config.CROSSBAR_IGNORE_CM
            self._sig_count = self._sig_count + 1 if label == self._sig_label else 1
            self._sig_label = label
            if self._sig_count < config.SIGNAL_CONFIRM_FRAMES or not self._sig_armed[label]:
                return

            if label == "PARE" and self.state in (State.SIGUIENDO, State.BUSCANDO):
                self._sig_armed["PARE"] = False
                self.state = State.PARE
                self.stop_until = now + self.stop_duration
                self.u = 0.0
                self.signal = "PARE"
                print(f"[CONTROL] Señal PARE detectada → detenido {self.stop_duration:.1f}s")
            elif label == "SIGA":
                self._sig_armed["SIGA"] = False
                self.signal = "SIGA"
                if self.state == State.PARE:
                    self.stop_until = now          # SIGA reanuda la marcha de inmediato
                print("[CONTROL] Señal SIGA detectada")

    def snapshot(self):
        """Copia del estado actual (para dibujarlo en pantalla sin tocar los datos reales)."""
        with self.lock:
            return {"state": self.state, "u": self.u, "error": self.error,
                    "line_found": self.line_found, "paused": self.paused,
                    "signal": self.signal, "nav_mode": self.nav.mode,
                    "stop_remaining": max(0.0, self.stop_until - self._now)}

    def map_snapshot(self):
        """Copia del mapa (para mapa.draw_map, que dibuja fuera del candado)."""
        with self.lock:
            return self.nav.snapshot()
