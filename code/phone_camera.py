#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  phone_camera.py — Servidor de Cámara Remota (iPhone/Android → PC)
=============================================================================

  Levanta DOS servidores:
    - HTTP  en puerto 8080: Página de guía con enlace al servidor HTTPS.
    - HTTPS en puerto 8443: Página con acceso a la cámara del celular.

  Safari en iPhone REQUIERE HTTPS para acceder a la cámara
  (navigator.mediaDevices.getUserMedia). Por eso se usan dos puertos:
  el HTTP siempre es accesible y guía al usuario hacia el HTTPS.

  Flujo:
    1. El usuario abre http://<IP>:8080 en Safari (siempre funciona).
    2. La página le muestra un enlace a https://<IP>:8443.
    3. Safari muestra advertencia de certificado → el usuario la acepta.
    4. La página HTTPS accede a la cámara y envía frames por POST.
    5. Python decodifica los frames y los expone vía cap.read().

  Compatible con: iPhone 8+ (Safari iOS 12+), Android (Chrome)

  Autor  : Equipo Visión Artificial
  Fecha  : 2026-09
  Licencia: MIT
=============================================================================
"""

import asyncio
import json
import logging
import os
import socket
import ssl
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("PhoneCamera")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(name)s] %(levelname)s: %(message)s"
    ))
    logger.addHandler(_handler)


# ─────────────────────────────────────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def get_local_ip():
    """Obtiene la IP local del PC en la red WiFi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─────────────────────────────────────────────────────────────────────────────
#  PÁGINA HTML DE GUÍA (servida en HTTP)
# ─────────────────────────────────────────────────────────────────────────────

def build_guide_html(local_ip, https_port):
    """Genera la página HTML de guía que redirige al servidor HTTPS."""
    https_url = f"https://{local_ip}:{https_port}"
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Seguidor de Línea — Configuración</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #111;
            color: #eee;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .card {{
            background: #1a1a2e;
            border-radius: 16px;
            padding: 30px;
            max-width: 400px;
            width: 100%;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }}
        h1 {{
            text-align: center;
            font-size: 22px;
            margin-bottom: 20px;
        }}
        .step {{
            background: #16213e;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 12px;
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }}
        .step-num {{
            background: #4caf50;
            color: #fff;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
            flex-shrink: 0;
        }}
        .step-text {{
            font-size: 14px;
            line-height: 1.5;
            color: #ccc;
        }}
        .step-text b {{ color: #fff; }}
        .big-btn {{
            display: block;
            width: 100%;
            padding: 16px;
            background: #4caf50;
            color: white;
            text-align: center;
            text-decoration: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 700;
            margin-top: 20px;
            transition: background 0.2s;
        }}
        .big-btn:active {{ background: #388e3c; }}
        .note {{
            text-align: center;
            font-size: 12px;
            color: #888;
            margin-top: 14px;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>📱 Conectar Cámara</h1>

        <div class="step">
            <span class="step-num">1</span>
            <span class="step-text">
                Toca el botón verde de abajo para ir al <b>servidor de cámara</b>.
            </span>
        </div>

        <div class="step">
            <span class="step-num">2</span>
            <span class="step-text">
                Safari mostrará <b>"Esta conexión no es privada"</b>.
                Toca <b>"Mostrar detalles"</b>.
            </span>
        </div>

        <div class="step">
            <span class="step-num">3</span>
            <span class="step-text">
                Toca <b>"visitar este sitio web"</b> y confirma.
            </span>
        </div>

        <div class="step">
            <span class="step-num">4</span>
            <span class="step-text">
                <b>Permite el acceso a la cámara</b> cuando Safari lo solicite.
            </span>
        </div>

        <div class="step">
            <span class="step-num">5</span>
            <span class="step-text">
                Presiona <b>"▶ Iniciar"</b> para comenzar a enviar video.
            </span>
        </div>

        <a class="big-btn" href="{https_url}">
            📷 Ir a la Cámara →
        </a>

        <p class="note">
            El certificado es auto-firmado (seguro, es tu red local).<br>
            Solo necesitas aceptarlo una vez.
        </p>
    </div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  PhoneCameraCapture — Drop-in replacement para cv2.VideoCapture
# ─────────────────────────────────────────────────────────────────────────────

class PhoneCameraCapture:
    """
    Servidor que recibe video de la cámara de un celular vía HTTP POST.

    Expone la misma interfaz básica que cv2.VideoCapture para integrarse
    sin cambios al pipeline existente de procesamiento de video.

    Arquitectura:
        - Puerto HTTP  (default 8080): Página de guía/instrucciones.
        - Puerto HTTPS (default 8443): Cámara + endpoint POST /frame.

    Uso típico::

        cap = PhoneCameraCapture(http_port=8080, https_port=8443)
        cap.start()
        # cap.read() funciona igual que cv2.VideoCapture
        ret, frame = cap.read()
        cap.release()
    """

    def __init__(self, http_port=8080, https_port=8443):
        """
        Args:
            http_port (int): Puerto del servidor HTTP (guía).
            https_port (int): Puerto del servidor HTTPS (cámara).
        """
        self.http_port = http_port
        self.https_port = https_port
        self.port = http_port  # Para compatibilidad con main.py
        self.local_ip = get_local_ip()

        # Frame buffer thread-safe
        self._lock = threading.Lock()
        self._frame = None
        self._frame_time = 0.0
        self._frame_count = 0
        self._connected = False

        # Control de servidores
        self._server_thread = None
        self._loop = None
        self._http_runner = None
        self._https_runner = None
        self._running = False

        # Estadísticas
        self._fps_counter = 0
        self._fps_time = time.time()
        self._current_fps = 0.0
        self._total_bytes = 0

        # Rutas
        self._template_dir = Path(__file__).parent / "templates"
        self._ssl_dir = Path(__file__).parent / ".ssl"

    # ── Interfaz compatible con cv2.VideoCapture ──────────────────────────

    def isOpened(self):
        """Retorna True si el servidor está corriendo."""
        return self._running

    def read(self):
        """
        Retorna el último frame recibido del celular.

        Returns:
            tuple[bool, np.ndarray | None]:
                (True, frame_bgr) si hay frame disponible.
                (False, None) si no hay frame aún.
        """
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
            return False, None

    def set(self, prop_id, value):
        """Stub de compatibilidad con cv2.VideoCapture.set()."""
        pass

    def get(self, prop_id):
        """Stub de compatibilidad con cv2.VideoCapture.get()."""
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            with self._lock:
                return self._frame.shape[1] if self._frame is not None else 320
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            with self._lock:
                return self._frame.shape[0] if self._frame is not None else 240
        elif prop_id == cv2.CAP_PROP_FPS:
            return self._current_fps
        return 0

    def release(self):
        """Detiene los servidores y libera recursos."""
        self._running = False
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=3.0)
        logger.info("Servidores detenidos.")

    # ── Inicio del servidor ──────────────────────────────────────────────

    def start(self):
        """Inicia ambos servidores (HTTP + HTTPS) en un hilo background."""
        if self._running:
            logger.warning("Los servidores ya están corriendo.")
            return

        self._running = True
        self._server_thread = threading.Thread(
            target=self._run_servers,
            name="PhoneCameraServer",
            daemon=True
        )
        self._server_thread.start()

        # Esperar a que el servidor esté listo
        timeout = 5.0
        start = time.time()
        while not self._loop and (time.time() - start) < timeout:
            time.sleep(0.05)

        self._print_connection_info()

    def _print_connection_info(self):
        """Muestra las instrucciones de conexión en la terminal."""
        http_url = f"http://{self.local_ip}:{self.http_port}"
        https_url = f"https://{self.local_ip}:{self.https_port}"
        print()
        print("=" * 60)
        print("  📱 SERVIDOR DE CÁMARA REMOTA LISTO")
        print("=" * 60)
        print()
        print(f"  Abre Safari en tu iPhone y ve a:")
        print()
        print(f"     👉  {http_url}")
        print()
        print(f"  La página te guiará paso a paso.")
        print()
        print(f"  (Servidor HTTPS en: {https_url})")
        print(f"  (Asegúrate de estar en la misma red WiFi)")
        print("=" * 60)
        print()

    # ── Generación de certificado SSL ────────────────────────────────────

    def _generate_ssl_cert(self):
        """
        Genera un certificado SSL auto-firmado compatible con iOS Safari.

        Returns:
            ssl.SSLContext: Contexto SSL configurado.
        """
        self._ssl_dir.mkdir(exist_ok=True)
        cert_file = self._ssl_dir / "cert.pem"
        key_file = self._ssl_dir / "key.pem"

        # Regenerar si la IP cambió (el cert está atado a la IP)
        ip_file = self._ssl_dir / "ip.txt"
        stored_ip = ip_file.read_text().strip() if ip_file.exists() else ""

        if not cert_file.exists() or not key_file.exists() or stored_ip != self.local_ip:
            logger.info("Generando certificado SSL auto-firmado...")

            # Crear archivo de configuración OpenSSL para SAN
            conf_file = self._ssl_dir / "openssl.cnf"
            conf_file.write_text(f"""[req]
default_bits = 2048
prompt = no
default_md = sha256
x509_extensions = v3_req
distinguished_name = dn

[dn]
CN = {self.local_ip}

[v3_req]
subjectAltName = IP:{self.local_ip}
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
""")

            try:
                subprocess.run([
                    "openssl", "req", "-x509",
                    "-newkey", "rsa:2048",
                    "-keyout", str(key_file),
                    "-out", str(cert_file),
                    "-days", "365",
                    "-nodes",
                    "-config", str(conf_file),
                ], check=True, capture_output=True, text=True)

                ip_file.write_text(self.local_ip)
                logger.info("✅ Certificado SSL generado correctamente.")
            except FileNotFoundError:
                logger.error(
                    "openssl no está instalado. "
                    "Instálalo con: sudo apt install openssl"
                )
                raise
            except subprocess.CalledProcessError as e:
                logger.error(f"Error generando certificado: {e.stderr}")
                raise
        else:
            logger.info("Usando certificado SSL existente.")

        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ssl_ctx.load_cert_chain(str(cert_file), str(key_file))
        return ssl_ctx

    # ── Servidores ───────────────────────────────────────────────────────

    def _run_servers(self):
        """Ejecuta ambos servidores (HTTP + HTTPS) en un event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # ── App HTTP (guía) ──
        http_app = web.Application()
        http_app.router.add_get("/", self._handle_guide)

        self._http_runner = web.AppRunner(http_app)
        self._loop.run_until_complete(self._http_runner.setup())

        http_site = web.TCPSite(self._http_runner, "0.0.0.0", self.http_port)
        self._loop.run_until_complete(http_site.start())
        logger.info(f"Servidor HTTP escuchando en 0.0.0.0:{self.http_port}")

        # ── App HTTPS (cámara) ──
        https_app = web.Application(client_max_size=10 * 1024 * 1024)  # 10 MB
        https_app.router.add_get("/", self._handle_camera_page)
        https_app.router.add_post("/frame", self._handle_frame_post)
        https_app.router.add_get("/ws", self._handle_websocket)

        self._https_runner = web.AppRunner(https_app)
        self._loop.run_until_complete(self._https_runner.setup())

        ssl_ctx = self._generate_ssl_cert()
        https_site = web.TCPSite(
            self._https_runner, "0.0.0.0", self.https_port,
            ssl_context=ssl_ctx
        )
        self._loop.run_until_complete(https_site.start())
        logger.info(f"Servidor HTTPS escuchando en 0.0.0.0:{self.https_port}")

        # Mantener el loop corriendo
        try:
            self._loop.run_forever()
        except Exception:
            pass
        finally:
            self._loop.run_until_complete(self._http_runner.cleanup())
            self._loop.run_until_complete(self._https_runner.cleanup())
            self._loop.close()

    async def _shutdown(self):
        """Detiene el event loop de forma segura."""
        if self._http_runner:
            await self._http_runner.cleanup()
        if self._https_runner:
            await self._https_runner.cleanup()
        self._loop.stop()

    # ── Handlers HTTP ────────────────────────────────────────────────────

    async def _handle_guide(self, request):
        """Sirve la página de guía (HTTP)."""
        html = build_guide_html(self.local_ip, self.https_port)
        return web.Response(text=html, content_type="text/html")

    async def _handle_camera_page(self, request):
        """Sirve la página de la cámara (HTTPS)."""
        html_path = self._template_dir / "camera.html"
        if not html_path.exists():
            return web.Response(
                text="Error: templates/camera.html no encontrado.",
                status=404
            )
        return web.FileResponse(html_path)

    async def _handle_frame_post(self, request):
        """
        Recibe un frame JPEG vía HTTP POST.

        El navegador envía el frame como body con Content-Type: image/jpeg.
        """
        try:
            jpeg_data = await request.read()
            if jpeg_data:
                self._decode_frame(jpeg_data)
                return web.Response(text="OK", status=200)
            return web.Response(text="No data", status=400)
        except Exception as e:
            logger.error(f"Error en POST /frame: {e}")
            return web.Response(text=str(e), status=500)

    async def _handle_websocket(self, request):
        """
        Maneja la conexión WebSocket con el celular (fallback).
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_ip = request.remote
        logger.info(f"📱 Cliente WebSocket conectado: {client_ip}")
        self._connected = True

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    self._decode_frame(msg.data)
                elif msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data.get("type") == "ping":
                            await ws.send_str(json.dumps({
                                "type": "pong",
                                "client_ts": data.get("client_ts", 0)
                            }))
                    except json.JSONDecodeError:
                        pass
                elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                    break
        except Exception as e:
            logger.error(f"Error en WebSocket: {e}")
        finally:
            self._connected = False
            logger.info(f"📱 Cliente WebSocket desconectado: {client_ip}")

        return ws

    # ── Decodificación de frames ─────────────────────────────────────────

    def _decode_frame(self, jpeg_data):
        """Decodifica un frame JPEG recibido."""
        try:
            np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.time()
                    self._frame_count += 1
                    self._connected = True

                # Estadísticas
                self._total_bytes += len(jpeg_data)
                self._fps_counter += 1
                now = time.time()
                elapsed = now - self._fps_time
                if elapsed >= 2.0:
                    self._current_fps = self._fps_counter / elapsed
                    bw = (self._total_bytes / elapsed) / 1024
                    logger.info(
                        f"📊 {self._current_fps:.1f} FPS | "
                        f"{bw:.0f} KB/s | "
                        f"Frame #{self._frame_count} | "
                        f"{frame.shape[1]}×{frame.shape[0]}"
                    )
                    self._fps_counter = 0
                    self._fps_time = now
                    self._total_bytes = 0
            else:
                logger.warning("Frame JPEG inválido recibido.")
        except Exception as e:
            logger.error(f"Error decodificando frame: {e}")

    # ── Propiedades de estado ────────────────────────────────────────────

    @property
    def is_connected(self):
        """¿Hay un celular conectado enviando frames?"""
        return self._connected

    @property
    def fps(self):
        """FPS actual del stream."""
        return self._current_fps

    @property
    def frame_count(self):
        """Total de frames recibidos."""
        return self._frame_count


# ─────────────────────────────────────────────────────────────────────────────
#  EJECUCIÓN INDEPENDIENTE (para testing)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PRUEBA DE CÁMARA REMOTA (modo standalone)")
    print("=" * 60)

    cap = PhoneCameraCapture(http_port=8080, https_port=8443)
    cap.start()

    print("Esperando conexión del celular...")
    print("Presiona 'q' en la ventana de video para salir.")

    try:
        while True:
            ret, frame = cap.read()

            if ret:
                h, w = frame.shape[:2]
                info = f"FPS: {cap.fps:.1f} | Frames: {cap.frame_count}"
                cv2.putText(frame, info, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                status = "CONECTADO" if cap.is_connected else "DESCONECTADO"
                color = (0, 255, 0) if cap.is_connected else (0, 0, 255)
                cv2.putText(frame, status, (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imshow("Phone Camera Test", frame)
            else:
                waiting = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(waiting, "Esperando camara...", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                cv2.putText(waiting, f"http://{cap.local_ip}:{cap.http_port}",
                            (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
                cv2.imshow("Phone Camera Test", waiting)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nCtrl+C detectado.")

    cap.release()
    cv2.destroyAllWindows()
    print("Fin de la prueba.")
