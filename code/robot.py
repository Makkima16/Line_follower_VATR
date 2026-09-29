"""
Conexión con el robot y el hilo que le envía los comandos.

``Robot`` solo transporta texto; no sabe qué significa. El significado lo
pone el driver (drivers.py). Así, cambiar de firmware no toca este archivo.
"""

import re            # Expresiones regulares para validar formato de MAC address
import socket        # Sockets Bluetooth (RFCOMM) para comunicarse con el robot
import threading     # Hilo independiente para enviar comandos sin bloquear el video

import config        # Parámetros: canal RFCOMM, baud rate, etc.

_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")  # Regex para validar MAC (ej: "00:1B:10:21:2C:1B")


class Robot:
    """
    Enlace con el robot.

    ``address``:
      * None → simulación (no se conecta nada).
      * MAC  → socket RFCOMM, igual que master_pc/Robot.py.
      * otro → puerto serie vía pyserial ("/dev/rfcomm0", "COM5").
    """

    def __init__(self, address, channel=config.RFCOMM_CHANNEL):  # Recibe dirección y canal Bluetooth
        self.address = address      # Dirección del robot: None, MAC o puerto serie
        self.channel = channel      # Canal RFCOMM (por defecto 1)
        self._sock = None           # Socket Bluetooth (si se conecta por MAC)
        self._serial = None         # Conexión serie (si se conecta por puerto)

    @property
    def simulado(self):             # Propiedad: ¿está en modo simulación?
        return self.address is None  # True si no se dio dirección

    def conectar(self):                                                          # Establece la conexión con el robot
        """Abre la conexión (Bluetooth por MAC o puerto serie). En simulación no hace nada."""
        if self.simulado:                                                        # Si no hay dirección
            print("[ROBOT] Simulación: no se envía nada, mira la ventana 'Movimiento'.")  # Avisa que es simulación
            return                                                               # No hace nada más
        print(f"[ROBOT] Conectando a {self.address}...")                         # Informa el intento de conexión
        if _MAC_RE.match(self.address):                                          # Si la dirección es una MAC válida
            self._sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM,  # Crea socket Bluetooth
                                       socket.BTPROTO_RFCOMM)                   # Protocolo RFCOMM (puerto serie BT)
            try:                                                                 # Intenta conectar
                self._sock.connect((self.address, self.channel))                 # Conexión al dispositivo por MAC y canal
            except OSError:                                                      # Si falla la conexión
                self.cerrar()                                                    # Cierra el socket
                raise                                                            # Re-lanza la excepción
            # CRÍTICO: sin timeout, si el robot deja de responder sendall() se
            # quedaría bloqueado para siempre y el hilo de envío nunca terminaría.
            self._sock.settimeout(0.5)                                           # Timeout de 500ms para que enviar no cuelgue
        else:                                                                    # Si no es MAC, asume puerto serie
            import serial                                                        # Importa pyserial (solo cuando se necesita)
            self._serial = serial.Serial(self.address, config.BAUD_RATE,         # Abre el puerto serie con el baud rate
                                         timeout=0, write_timeout=0.5)           # Lectura no bloqueante, escritura con timeout
        print("[ROBOT] Conexión establecida.")                                   # Confirma conexión exitosa

    def enviar(self, mensaje):                    # Envía un mensaje de texto al robot
        """
        Envía un mensaje corto. No hace ``sleep``: los tiempos los decide
        el hilo de envío. (El ejemplo del curso dormía 0.1 s aquí.)
        """
        data = mensaje.encode("ascii")            # Convierte el texto a bytes ASCII
        if self._sock is not None:                # Si está conectado por Bluetooth
            self._sock.sendall(data)              # Envía todos los bytes por el socket
        elif self._serial is not None:            # Si está conectado por puerto serie
            self._serial.write(data)              # Escribe los bytes al puerto serie

    def vaciar_respuestas(self):                  # Lee y descarta respuestas del firmware
        """
        Lee y descarta lo que responde el firmware ("Comando: Adelante"...).

        Si nadie lo lee, el buffer se llena, el Bluetooth aplica control de
        flujo y el ``Serial.println`` del Arduino termina trabando el robot.
        """
        try:                                                      # Intenta leer sin bloquear
            if self._sock is not None:                            # Si hay socket Bluetooth
                self._sock.setblocking(False)                     # Modo no bloqueante para leer
                try:                                              # Intenta leer todo lo que haya
                    while self._sock.recv(256):                   # Lee en bloques de 256 bytes hasta que no haya más
                        pass                                      # Descarta lo leído
                finally:                                          # Siempre restaura el timeout original
                    self._sock.settimeout(0.5)                    # Vuelve a modo con timeout de 500ms
            elif self._serial is not None and self._serial.in_waiting:  # Si hay serie y tiene datos pendientes
                self._serial.read(self._serial.in_waiting)        # Lee y descarta todos los bytes en el buffer
        except (BlockingIOError, InterruptedError):               # Si no hay datos disponibles
            pass                                                  # Es normal, no hacer nada

    def cerrar(self):                                  # Cierra la conexión con el robot
        """Cierra socket/puerto serie. Se puede llamar varias veces sin error."""
        for conn in (self._sock, self._serial):        # Recorre socket y serial
            if conn is not None:                       # Si la conexión existe
                try:                                   # Intenta cerrar
                    conn.close()                       # Cierra la conexión
                except OSError:                        # Si falla al cerrar
                    pass                               # Ignora el error (ya estaba cerrado o no importa)
                print("[ROBOT] Conexión cerrada.")     # Informa el cierre
        self._sock = None                              # Limpia la referencia al socket
        self._serial = None                            # Limpia la referencia al serial


class CommandSender(threading.Thread):
    """
    Hilo que alimenta al robot al ritmo que el driver indique.

    En cada vuelta pide ``brain.next_command()`` → ``(mensaje, espera, mov)``,
    envía el mensaje (si hay), informa el movimiento al simulador y espera.
    La cámara nunca espera al Bluetooth ni el Bluetooth a la cámara.
    """

    def __init__(self, robot, brain, on_motion=None):          # Recibe robot, cerebro y callback de movimiento
        # CRÍTICO: daemon=True hace que el programa pueda cerrarse aunque este
        # hilo siga vivo; sin él, un cierre inesperado dejaría el proceso colgado.
        super().__init__(name="CommandSender", daemon=True)     # Thread daemon: se cierra cuando el programa termina
        self.robot = robot                                     # Referencia al objeto Robot (conexión)
        self.brain = brain                                     # Referencia al Brain (control PD + estados)
        self.on_motion = on_motion                             # Callback opcional para reportar movimiento (al simulador)
        self.last_message = None                               # Último mensaje enviado (para mostrarlo en pantalla)
        self.error = None                                      # Error de conexión (None = todo bien)
        self._stop_event = threading.Event()                   # Evento para señalar que el hilo debe terminar

    def run(self):                                                      # Método principal del hilo (se ejecuta en paralelo)
        """Bucle del hilo: pedir comando → enviarlo → vaciar respuestas → esperar."""
        try:                                                            # Bloque principal con manejo de errores
            while not self._stop_event.is_set():                        # Mientras no se pida detener
                mensaje, espera, motion = self.brain.next_command()     # Pide al cerebro el siguiente comando
                if mensaje is not None:                                 # Si hay un mensaje que enviar
                    self.robot.enviar(mensaje)                          # Lo envía al robot por BT o serie
                    self.last_message = mensaje                         # Guarda el último mensaje (para debug)
                if self.on_motion is not None:                          # Si hay callback de movimiento
                    self.on_motion(motion, mensaje)                     # Reporta el movimiento al simulador visual
                # CRÍTICO: si no se vacían las respuestas, el buffer se llena y el
                # robot termina trabándose (ver vaciar_respuestas).
                self.robot.vaciar_respuestas()                          # Vacía el buffer de respuestas del firmware
                self._stop_event.wait(espera)                           # Espera el tiempo indicado por el driver (interruptible)
        except OSError as e:                                            # Si se pierde la conexión Bluetooth/serie
            self.error = e                                              # Guarda el error para que main.py lo detecte
            print(f"[ROBOT] Se perdió la conexión: {e}")                # Informa por consola
        finally:                                                        # Siempre, al terminar el hilo
            # CRÍTICO: sin este envío final el robot se quedaría ejecutando el
            # último comando (seguiría andando) después de cerrar el programa.
            try:                                                        # Intenta enviar la orden de parada
                self.robot.enviar(self.brain.driver.stop_message())     # Envía 'x' o "V:0,0\n" para detener el robot
            except OSError:                                             # Si no se puede enviar
                pass                                                    # No pasa nada, ya se perdió la conexión

    def detener(self):                    # Solicita al hilo que termine
        """Pide al hilo que termine y espera (máx. 1 s) a que mande la parada."""
        self._stop_event.set()            # Activa el evento de parada (desbloquea el wait)
        self.join(timeout=1.0)            # Espera hasta 1 segundo a que el hilo termine
