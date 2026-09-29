#!/usr/bin/env python3
"""
Robot seguidor de línea: punto de entrada.

Hilos:
  * Principal: cámara → visión → Brain.on_frame() → ventanas de OpenCV.
  * CommandSender: Brain.next_command() → driver → robot (o simulador).

Uso (por defecto: cámara del celular y robot simulado):
  python main.py
  python main.py --camara usb
  python main.py --video pista.mp4
  python main.py --robot 00:1B:10:21:2C:1B            # mBot real, firmware de pulsos
  python main.py --robot /dev/rfcomm0 --driver velocidades

Ventanas: "Camara" (detección), "Mapa (odometria)" (cruces y ramas recordadas),
"Movimiento" (comandos enviados) y "Calibracion" (trackbars).

Teclas: q = salir, espacio = pausar/reanudar, r = reiniciar trayectoria simulada.
"""

import argparse
import time

import cv2
import numpy as np

import config
import mapa
import vision
import senales
from control import Brain, State
from drivers import DRIVERS, make_driver
from robot import CommandSender, Robot
from sim import MotionPreview

# Nombres de las ventanas de OpenCV (se usan como identificadores; deben coincidir)
WIN_VIEW = "Camara"
WIN_MASK = "Linea (binaria)"
WIN_TUNE = "Calibracion"
WIN_SIM = "Movimiento"
WIN_MAP = "Mapa (odometria)"


# ─────────────────────────────────────────────────────────────────────────────
#  TRACKBARS
# ─────────────────────────────────────────────────────────────────────────────
def create_trackbars(driver):
    """
    Crea la ventana "Calibracion" con los sliders (umbral, ROI, Kp, Kd, velocidad...).
    Los valores se guardan x100 porque los trackbars solo manejan enteros.
    CRÍTICO: debe llamarse antes de read_trackbars(); si la ventana no existe,
    getTrackbarPos no encuentra los sliders.
    """
    cv2.namedWindow(WIN_TUNE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_TUNE, 420, 300)
    nop = lambda _v: None
    cv2.createTrackbar("Umbral (0=Otsu)", WIN_TUNE, config.BINARY_THRESHOLD, 255, nop)
    cv2.createTrackbar("ROI %", WIN_TUNE, int(config.ROI_HEIGHT_RATIO * 100), 100, nop)
    cv2.createTrackbar("Peso cercano %", WIN_TUNE, int(config.NEAR_WEIGHT * 100), 100, nop)
    cv2.createTrackbar("Kp x100", WIN_TUNE, int(config.KP * 100), 400, nop)
    cv2.createTrackbar("Kd x100", WIN_TUNE, int(config.KD * 100), 200, nop)
    cv2.createTrackbar("Zona muerta x100", WIN_TUNE, int(config.DEADBAND * 100), 50, nop)
    cv2.createTrackbar("Giros max", WIN_TUNE, config.MAX_TURN_PULSES, 10, nop)
    cv2.createTrackbar("Acelerador %", WIN_TUNE, int(config.FORWARD_DUTY * 100), 100, nop)
    cv2.createTrackbar("Vel. base", WIN_TUNE, config.BASE_SPEED, config.MAX_SPEED, nop)
    cv2.createTrackbar("Dur. PARE [s]", WIN_TUNE, int(config.STOP_DURATION_S), 10, nop)


def read_trackbars(brain):
    """
    Aplica los sliders al control y devuelve los parámetros de visión.
    Se llama en cada frame: así se calibra en vivo sin reiniciar el programa.
    Devuelve (umbral, roi_ratio, near_weight).
    """
    tb = lambda name: cv2.getTrackbarPos(name, WIN_TUNE)
    driver = brain.driver
    # Con el candado: el hilo de envío lee estos valores al mismo tiempo
    with brain.lock:
        brain.pd.kp = tb("Kp x100") / 100.0
        brain.pd.kd = tb("Kd x100") / 100.0
        driver.deadband = tb("Zona muerta x100") / 100.0
        if driver.name == "pulsos":
            driver.max_turn_pulses = max(1, tb("Giros max"))
            driver.forward_duty = tb("Acelerador %") / 100.0
        else:
            driver.base = tb("Vel. base")
        brain.stop_duration = float(tb("Dur. PARE [s]"))
    return (tb("Umbral (0=Otsu)"),
            max(tb("ROI %"), 10) / 100.0,
            tb("Peso cercano %") / 100.0)


# ─────────────────────────────────────────────────────────────────────────────
#  OVERLAY DE DEPURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
# Color (BGR) con que se escribe cada estado en pantalla
CMD_COLORS = {
    "ADELANTE":  (0, 210, 0),
    "IZQUIERDA": (0, 210, 255),
    "DERECHA":   (255, 150, 0),
    "ATRAS":     (220, 0, 220),
    "PARAR":     (0, 0, 255),
}

STATE_COLORS = {
    State.SIGUIENDO: (0, 200, 0),
    State.BUSCANDO: (0, 140, 255),
    State.DETENIDO: (0, 0, 255),
    State.PARE: (0, 0, 220),
    State.MEDIA_VUELTA: (255, 0, 255),
    State.FIN: (200, 200, 200),
}


def put_text(img, text, org, color=(255, 255, 255), scale=0.55):
    """Escribe texto con un fondo negro detrás para que se lea sobre cualquier imagen."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(frame, line, roi_ratio, snap, last_msg, fps, scale, signal=None,
                 driver=None):
    """
    Dibuja la información de depuración sobre una copia reescalada del frame:
    límite de la ROI, contornos y centroides de la línea, señal detectada,
    barra del mando u, estado, error, último comando y FPS.

    snap es brain.snapshot(); scale solo cambia el tamaño de la ventana.
    """
    h, w = frame.shape[:2]
    view = cv2.resize(frame, (int(w * scale), int(h * scale)))
    vh, vw = view.shape[:2]

    roi_y = int(h * (1.0 - roi_ratio) * scale)
    cv2.line(view, (0, roi_y), (vw, roi_y), (255, 200, 0), 1)
    cv2.line(view, (vw // 2, roi_y), (vw // 2, vh), (120, 120, 120), 1)   # Centro

    if line is not None:
        for c in line.contours:
            cv2.drawContours(view, [(c * scale).astype(int)], -1, (0, 255, 0), 2)
        pts = [(int(x * scale), int(y * scale)) for x, y in line.points]
        for p in pts:
            cv2.circle(view, p, 5, (0, 0, 255), -1)
        for a, b in zip(pts, pts[1:]):
            cv2.line(view, a, b, (0, 0, 255), 2)
        # Salidas por el borde de la ROI: 2+ = ramificación (amarillo), 1 = normal (cian)
        exit_color = (0, 255, 255) if len(line.exits) >= 2 else (255, 255, 0)
        for e in line.exits:
            cv2.circle(view, (int(e.px[0] * scale), int(e.px[1] * scale)), 8, exit_color, 2)

    # ── Señal de tráfico detectada ────────────────────────────────────────
    if signal is not None:
        sig_color = (0, 0, 255) if signal.label == "PARE" else (0, 200, 0)
        # Contorno del octágono
        scaled_cnt = (signal.contour * scale).astype(int)
        cv2.drawContours(view, [scaled_cnt], -1, sig_color, 3)
        # Bounding box
        sx, sy, sw, sh = signal.bbox
        sx, sy = int(sx * scale), int(sy * scale)
        sw, sh = int(sw * scale), int(sh * scale)
        cv2.rectangle(view, (sx, sy), (sx + sw, sy + sh), sig_color, 2)
        # Etiqueta sobre el bounding box
        put_text(view, signal.label + (" (parcial)" if signal.partial else ""),
                 (sx, sy - 8), sig_color, 0.7)
        # Borde de la ventana del color de la señal
        cv2.rectangle(view, (0, 0), (vw - 1, vh - 1), sig_color, 4)

    # ── Cuenta regresiva PARE ─────────────────────────────────────────────
    if snap["state"] == State.PARE:
        remaining = snap.get("stop_remaining", 0)
        pare_txt = f"PARE - Detenido {remaining:.1f}s"
        put_text(view, pare_txt, (vw // 2 - 100, vh // 2), (0, 0, 255), 0.8)
        # Borde rojo parpadeante
        if int(time.monotonic() * 4) % 2 == 0:
            cv2.rectangle(view, (0, 0), (vw - 1, vh - 1), (0, 0, 255), 6)

    # Barra del mando u: centro = recto, extremos = giro máximo
    cx, by = vw // 2, vh - 18
    cv2.rectangle(view, (20, by - 6), (vw - 20, by + 6), (60, 60, 60), -1)
    cv2.rectangle(view, (cx, by - 6), (cx + int(snap["u"] * (cx - 20)), by + 6),
                  (0, 200, 255), -1)
    cv2.line(view, (cx, by - 10), (cx, by + 10), (255, 255, 255), 1)

    state = snap["state"]
    label = "PAUSA" if snap["paused"] else state.value
    put_text(view, f"{label} | {snap['nav_mode']}", (10, 24),
             STATE_COLORS.get(state, (255, 255, 255)), 0.7)
    put_text(view, f"e={snap['error']:+.2f}  u={snap['u']:+.2f}", (10, 50))
    cmd = driver.describe(last_msg) if driver is not None else "-"
    put_text(view, cmd, (10, 78), CMD_COLORS.get(cmd, (255, 255, 255)), 0.8)
    # Mostrar señal activa en el HUD
    sig_label = snap.get("signal")
    if sig_label:
        sig_c = (0, 0, 255) if sig_label == "PARE" else (0, 200, 0)
        put_text(view, f"Senal: {sig_label}", (10, 104), sig_c, 0.55)
    put_text(view, f"{fps:4.1f} FPS", (vw - 95, 24))
    return view


# ─────────────────────────────────────────────────────────────────────────────
#  CÁMARA
# ─────────────────────────────────────────────────────────────────────────────
def open_camera(args):
    """Abre la fuente de video según los argumentos: archivo, celular o cámara USB."""
    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif args.camara == "phone":
        from phone_camera import PhoneCameraCapture   # Necesita aiohttp
        cap = PhoneCameraCapture(http_port=config.PHONE_HTTP_PORT,
                                 https_port=config.PHONE_HTTPS_PORT)
        cap.start()
        return cap
    else:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        # Buffer de 1 frame: siempre se procesa la imagen más reciente
        # (sin esto la cámara acumula frames viejos y el robot reacciona tarde)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # CRÍTICO: sin esta comprobación una cámara inexistente no daría error aquí
    # y el programa se quedaría leyendo frames vacíos.
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara/video")
    return cap


def waiting_screen(text):
    """Imagen gris con un mensaje, para mostrar mientras no llega video."""
    img = np.full((240, 480, 3), 30, np.uint8)
    put_text(img, text, (20, 120), (0, 200, 255))
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def run(cap, brain, sender, preview, args):
    """
    Bucle principal (hilo de video). En cada vuelta:
      leer frame → preparar → detectar línea y señal → actualizar Brain → dibujar.
    El envío al robot lo hace aparte el hilo CommandSender.
    """
    prev_x = None
    last_frame_id = None
    fps, n, t0 = 0.0, 0, time.monotonic()

    while True:
        # Si el hilo de envío perdió la conexión, no tiene sentido seguir
        if sender.error is not None:
            print("[MAIN] Sin conexión con el robot, se termina.")
            break

        cv2.imshow(WIN_SIM, preview.draw())
        cv2.imshow(WIN_MAP, mapa.draw_map(brain.map_snapshot()))

        ok, frame = cap.read()
        if not ok:
            if args.video:
                break
            cv2.imshow(WIN_VIEW, waiting_screen("Esperando video del celular..."))
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                break
            continue

        # La cámara del celular devuelve el último frame; no repetir el mismo
        frame_id = getattr(cap, "frame_count", None)
        if frame_id is not None:
            if frame_id == last_frame_id:
                key = cv2.waitKey(5) & 0xFF
                if key == ord("q"):
                    break
                continue
            last_frame_id = frame_id

        frame = vision.prepare_frame(frame)
        threshold, roi_ratio, near_weight = read_trackbars(brain)
        line = vision.detect_line(frame, threshold, roi_ratio, near_weight, prev_x)
        # Recordar dónde estaba la línea para no saltar a otra mancha en el siguiente frame
        prev_x = line.points[0][0] if line is not None else None

        # Detección de señales de tráfico (objetivos 5 y 6). Se informa en cada
        # frame, también cuando no hay señal (para confirmar y re-armar).
        signal = senales.detect_signal(frame)
        brain.on_signal(signal.label if signal is not None else None, time.monotonic(),
                        partial=signal is not None and signal.partial)

        # La máquina de estados (incluido el PARE) vive en Brain, bajo su candado
        brain.on_frame(line)

        # Cálculo de FPS: frames contados / segundos transcurridos (cada ~1 s)
        n += 1
        now = time.monotonic()
        if now - t0 >= 1.0:
            fps, n, t0 = n / (now - t0), 0, now

        cv2.imshow(WIN_VIEW, draw_overlay(frame, line, roi_ratio, brain.snapshot(),
                                          sender.last_message, fps, args.escala,
                                          signal, brain.driver))
        # CRÍTICO: si no hay línea, line es None y line.binary daría AttributeError
        if line is not None:
            cv2.imshow(WIN_MASK, line.binary)

        # CRÍTICO: waitKey es lo que refresca las ventanas de OpenCV y lee el
        # teclado; sin él las ventanas se quedan congeladas.
        key = cv2.waitKey(1 if not args.video else 30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            brain.set_paused(not brain.snapshot()["paused"])
        if key == ord("r"):
            preview.reset()


def parse_args():
    """Lee las opciones de la línea de comandos (--camara, --video, --robot, --driver, --escala)."""
    p = argparse.ArgumentParser(description="Robot seguidor de línea")
    p.add_argument("--camara", choices=("phone", "usb"), default=config.CAMERA_SOURCE)
    p.add_argument("--video", help="Procesar un archivo de video en vez de la cámara")
    p.add_argument("--robot", default=config.ROBOT_ADDRESS,
                   help="MAC o puerto serie del robot (sin esto: simulación)")
    p.add_argument("--driver", choices=list(DRIVERS), default=config.DRIVER)
    p.add_argument("--escala", type=float, default=config.DISPLAY_SCALE)
    return p.parse_args()


def main():
    """Crea las piezas (driver, Brain, Robot, vista), conecta, arranca el hilo de envío y el bucle."""
    args = parse_args()
    driver = make_driver(args.driver)
    brain = Brain(driver)
    robot = Robot(args.robot)
    preview = MotionPreview()
    cap = None
    sender = None

    print(f"[MAIN] Cámara: {args.video or args.camara} | Driver: {driver.name} | "
          f"Robot: {args.robot or 'simulado'}")
    try:
        cap = open_camera(args)
        robot.conectar()
        create_trackbars(driver)
        sender = CommandSender(robot, brain, on_motion=preview.add)
        sender.start()
        print("[MAIN] En marcha. q = salir, espacio = pausa, r = reiniciar trayectoria.")
        run(cap, brain, sender, preview, args)
    except KeyboardInterrupt:
        pass
    finally:
        # CRÍTICO: el bloque finally se ejecuta siempre (incluso con error o
        # Ctrl+C); garantiza que el robot se detenga y se liberen cámara y conexión.
        if sender is not None:
            sender.detener()          # Envía el mensaje de parada antes de salir
        robot.cerrar()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


# CRÍTICO: simulador.py importa funciones de este archivo; esta condición evita
# que al importarlo se ejecute main() y se abra la cámara.
if __name__ == "__main__":
    main()
