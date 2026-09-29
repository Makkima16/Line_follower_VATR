"""
Conexión con el robot y el hilo que le envía los comandos.

``Robot`` solo transporta texto; no sabe qué significa. El significado lo
pone el driver (drivers.py). Así, cambiar de firmware no toca este archivo.
"""

import re
import socket
import threading

import config

_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


class Robot:
    """
    Enlace con el robot.

    ``address``:
      * None → simulación (no se conecta nada).
      * MAC  → socket RFCOMM, igual que master_pc/Robot.py.
      * otro → puerto serie vía pyserial ("/dev/rfcomm0", "COM5").
    """

    def __init__(self, address, channel=config.RFCOMM_CHANNEL):
        self.address = address
        self.channel = channel
        self._sock = None
        self._serial = None

    @property
    def simulado(self):
        return self.address is None

    def conectar(self):
        if self.simulado:
            print("[ROBOT] Simulación: no se envía nada, mira la ventana 'Movimiento'.")
            return
        print(f"[ROBOT] Conectando a {self.address}...")
        if _MAC_RE.match(self.address):
            self._sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM,
                                       socket.BTPROTO_RFCOMM)
            try:
                self._sock.connect((self.address, self.channel))
            except OSError:
                self.cerrar()
                raise
            self._sock.settimeout(0.5)    # Que un envío nunca cuelgue el hilo
        else:
            import serial
            self._serial = serial.Serial(self.address, config.BAUD_RATE,
                                         timeout=0, write_timeout=0.5)
        print("[ROBOT] Conexión establecida.")

    def enviar(self, mensaje):
        """
        Envía un mensaje corto. No hace ``sleep``: los tiempos los decide
        el hilo de envío. (El ejemplo del curso dormía 0.1 s aquí.)
        """
        data = mensaje.encode("ascii")
        if self._sock is not None:
            self._sock.sendall(data)
        elif self._serial is not None:
            self._serial.write(data)

    def vaciar_respuestas(self):
        """
        Lee y descarta lo que responde el firmware ("Comando: Adelante"...).

        Si nadie lo lee, el buffer se llena, el Bluetooth aplica control de
        flujo y el ``Serial.println`` del Arduino termina trabando el robot.
        """
        try:
            if self._sock is not None:
                self._sock.setblocking(False)
                try:
                    while self._sock.recv(256):
                        pass
                finally:
                    self._sock.settimeout(0.5)
            elif self._serial is not None and self._serial.in_waiting:
                self._serial.read(self._serial.in_waiting)
        except (BlockingIOError, InterruptedError):
            pass

    def cerrar(self):
        for conn in (self._sock, self._serial):
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
                print("[ROBOT] Conexión cerrada.")
        self._sock = None
        self._serial = None


class CommandSender(threading.Thread):
    """
    Hilo que alimenta al robot al ritmo que el driver indique.

    En cada vuelta pide ``brain.next_command()`` → ``(mensaje, espera, mov)``,
    envía el mensaje (si hay), informa el movimiento al simulador y espera.
    La cámara nunca espera al Bluetooth ni el Bluetooth a la cámara.
    """

    def __init__(self, robot, brain, on_motion=None):
        super().__init__(name="CommandSender", daemon=True)
        self.robot = robot
        self.brain = brain
        self.on_motion = on_motion
        self.last_message = None
        self.error = None
        self._stop_event = threading.Event()

    def run(self):
        try:
            while not self._stop_event.is_set():
                mensaje, espera, motion = self.brain.next_command()
                if mensaje is not None:
                    self.robot.enviar(mensaje)
                    self.last_message = mensaje
                if self.on_motion is not None:
                    self.on_motion(motion, mensaje)
                self.robot.vaciar_respuestas()
                self._stop_event.wait(espera)
        except OSError as e:
            self.error = e
            print(f"[ROBOT] Se perdió la conexión: {e}")
        finally:
            try:
                self.robot.enviar(self.brain.driver.stop_message())
            except OSError:
                pass

    def detener(self):
        self._stop_event.set()
        self.join(timeout=1.0)
