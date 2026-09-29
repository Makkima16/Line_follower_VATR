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

Teclas: q = salir, espacio = pausar/reanudar, r = reiniciar trayectoria simulada.
"""

import argparse
import time

import cv2
import numpy as np

import config
import vision
import senales
from control import Brain, State
from drivers import DRIVERS, make_driver
from robot import CommandSender, Robot
from sim import MotionPreview

WIN_VIEW = "Camara"
WIN_MASK = "Linea (binaria)"
WIN_TUNE = "Calibracion"
WIN_SIM = "Movimiento"


# ─────────────────────────────────────────────────────────────────────────────
#  TRACKBARS
# ─────────────────────────────────────────────────────────────────────────────
def create_trackbars(driver):
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
    cv2.createTrackbar("Vel. base", WIN_TUNE, config.BASE_SPEED, config.MAX_SPEED, nop)
    cv2.createTrackbar("Dur. PARE [s]", WIN_TUNE, int(config.STOP_DURATION_S), 10, nop)


def read_trackbars(brain):
    """Aplica los sliders al control y devuelve los parámetros de visión."""
    tb = lambda name: cv2.getTrackbarPos(name, WIN_TUNE)
    driver = brain.driver
    with brain.lock:
        brain.pd.kp = tb("Kp x100") / 100.0
        brain.pd.kd = tb("Kd x100") / 100.0
        driver.deadband = tb("Zona muerta x100") / 100.0
        if driver.name == "pulsos":
            driver.max_turn_pulses = max(1, tb("Giros max"))
        else:
            driver.base = tb("Vel. base")
    return (tb("Umbral (0=Otsu)"),
            max(tb("ROI %"), 10) / 100.0,
            tb("Peso cercano %") / 100.0)


# ─────────────────────────────────────────────────────────────────────────────
#  OVERLAY DE DEPURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
STATE_COLORS = {
    State.SIGUIENDO: (0, 200, 0),
    State.BUSCANDO: (0, 140, 255),
    State.DETENIDO: (0, 0, 255),
}


def put_text(img, text, org, color=(255, 255, 255), scale=0.55):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(frame, line, roi_ratio, snap, last_msg, fps, scale):
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

    # Barra del mando u: centro = recto, extremos = giro máximo
    cx, by = vw // 2, vh - 18
    cv2.rectangle(view, (20, by - 6), (vw - 20, by + 6), (60, 60, 60), -1)
    cv2.rectangle(view, (cx, by - 6), (cx + int(snap["u"] * (cx - 20)), by + 6),
                  (0, 200, 255), -1)
    cv2.line(view, (cx, by - 10), (cx, by + 10), (255, 255, 255), 1)

    state = snap["state"]
    label = "PAUSA" if snap["paused"] else state.value
    put_text(view, label, (10, 24), STATE_COLORS[state], 0.7)
    msg = (last_msg or "-").strip()
    put_text(view, f"e={snap['error']:+.2f}  u={snap['u']:+.2f}  cmd={msg}", (10, 50))
    put_text(view, f"{fps:4.1f} FPS", (vw - 95, 24))
    return view


# ─────────────────────────────────────────────────────────────────────────────
#  CÁMARA
# ─────────────────────────────────────────────────────────────────────────────
def open_camera(args):
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara/video")
    return cap


def waiting_screen(text):
    img = np.full((240, 480, 3), 30, np.uint8)
    put_text(img, text, (20, 120), (0, 200, 255))
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def run(cap, brain, sender, preview, args):
    prev_x = None
    last_frame_id = None
    fps, n, t0 = 0.0, 0, time.monotonic()

    while True:
        if sender.error is not None:
            print("[MAIN] Sin conexión con el robot, se termina.")
            break

        cv2.imshow(WIN_SIM, preview.draw())

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
        prev_x = line.points[0][0] if line is not None else None

        # Detección de señales de tráfico (objetivos 5 y 6)
        signal = senales.detect_signal(frame)
        if signal is not None:
            brain.on_signal(signal.label, time.monotonic())

        brain.on_frame(line)

        n += 1
        now = time.monotonic()
        if now - t0 >= 1.0:
            fps, n, t0 = n / (now - t0), 0, now

        cv2.imshow(WIN_VIEW, draw_overlay(frame, line, roi_ratio, brain.snapshot(),
                                          sender.last_message, fps, args.escala))
        if line is not None:
            cv2.imshow(WIN_MASK, line.binary)

        key = cv2.waitKey(1 if not args.video else 30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            brain.set_paused(not brain.snapshot()["paused"])
        if key == ord("r"):
            preview.reset()

        # Manejo del estado PARE: detener el robot el tiempo que fije el docente
        if brain.state == State.PARE:
            # Leer duración actual de la trackbar (0-10 segundos)
            config.STOP_DURATION_S = cv2.getTrackbarPos("Dur. PARE [s]", WIN_TUNE) / 10.0
            brain.u = 0.0  # Garantizar que no haya movimiento
            # Mostrar "PARE" en el overlay
            cv2.putText(frame, "PARE - Deteniendo {:.1f}s".format(
                        max(0, brain.stop_until - time.monotonic())),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Si ya venció el tiempo, reanudar automáticamente
            if time.monotonic() >= brain.stop_until:
                brain.state = State.SIGUIENDO
                print("[MAIN] Reanudando marcha automáticamente después de PARE")

        # Mostrar señal de tráfico detectada (solo texto, no afecta control)
        sig = brain.snapshot().get("signal", None)
        if sig:
            cv2.putText(frame, "Senal: {}".format(sig),
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def parse_args():
    p = argparse.ArgumentParser(description="Robot seguidor de línea")
    p.add_argument("--camara", choices=("phone", "usb"), default=config.CAMERA_SOURCE)
    p.add_argument("--video", help="Procesar un archivo de video en vez de la cámara")
    p.add_argument("--robot", default=config.ROBOT_ADDRESS,
                   help="MAC o puerto serie del robot (sin esto: simulación)")
    p.add_argument("--driver", choices=list(DRIVERS), default=config.DRIVER)
    p.add_argument("--escala", type=float, default=config.DISPLAY_SCALE)
    return p.parse_args()


def main():
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
        if sender is not None:
            sender.detener()          # Envía el mensaje de parada antes de salir
        robot.cerrar()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
