import socket
import time


class Robot:
    def __init__(self, mac_address: str, port: int = 1):
        self.mac_address = mac_address
        self.port = port
        self.bluetooth_socket = None

    def conectar(self):
        self.bluetooth_socket = socket.socket(
            socket.AF_BLUETOOTH,
            socket.SOCK_STREAM,
            socket.BTPROTO_RFCOMM
        )

        try:
            print(f"Conectando a {self.mac_address}...")
            self.bluetooth_socket.connect((self.mac_address, self.port))
            print("Conexión establecida")
        except OSError:
            self.cerrar()
            raise

    def _enviar(self, comando: str):
        if self.bluetooth_socket is None:
            raise RuntimeError("El robot no está conectado")

        self.bluetooth_socket.sendall(comando.encode("utf-8"))
        print(f"Comando enviado: {comando}")
        time.sleep(0.1)

    def adelante(self):
        self._enviar("w")

    def atras(self):
        self._enviar("s")

    def izquierda(self):
        self._enviar("a")

    def derecha(self):
        self._enviar("d")

    def parar(self):
        self._enviar("x")

    def cerrar(self):
        if self.bluetooth_socket is not None:
            self.bluetooth_socket.close()
            self.bluetooth_socket = None
            print("Conexión cerrada")