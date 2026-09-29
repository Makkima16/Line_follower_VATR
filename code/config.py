"""
Parámetros del seguidor de línea.

Cuando llegue el robot, normalmente solo hay que tocar la sección
"ROBOT" de este archivo. Los marcados con [TB] también se ajustan en vivo
desde la ventana de trackbars "Calibracion".
"""

# ─────────────────────────────────────────────────────────────────────────────
#  CÁMARA
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_SOURCE = "phone"             # "phone" (celular por WiFi) o "usb"
CAMERA_INDEX = 0                    # Solo para "usb"
PHONE_HTTP_PORT = 8080
PHONE_HTTPS_PORT = 8443

# Según cómo quede montado el celular en el robot:
ROTATE = 0                          # 0, 90, 180 o 270 grados (sentido horario)
FLIP_HORIZONTAL = False             # Espejo izquierda/derecha

PROCESS_WIDTH = 320                 # Se reescala a este ancho (se conserva la proporción)
DISPLAY_SCALE = 2.0                 # Ampliación de la ventana de depuración

# ─────────────────────────────────────────────────────────────────────────────
#  DETECCIÓN DE LA LÍNEA (negra sobre fondo claro)
# ─────────────────────────────────────────────────────────────────────────────
ROI_HEIGHT_RATIO = 0.60             # [TB] Se analiza este % inferior del frame
N_SLICES = 5                        # Franjas horizontales dentro de la ROI
MIN_BLOB_AREA_RATIO = 0.01          # Área mínima de un trozo de línea (fracción de la franja)
MAX_JUMP_RATIO = 0.25               # Salto horizontal máx. entre franjas (fracción del ancho)
NEAR_WEIGHT = 0.6                   # [TB] Peso del punto cercano frente al lejano en el error
BINARY_THRESHOLD = 80               # [TB] 0 = Otsu automático

# ─────────────────────────────────────────────────────────────────────────────
#  CONTROL PD
# ─────────────────────────────────────────────────────────────────────────────
# El error se normaliza a [-1, 1] (±1 = línea en el borde de la imagen) y la
# salida u también: u = +1 "gira todo a la derecha", u = -1 "todo a la
# izquierda". Así el control no depende ni de la cámara ni del robot.
KP = 1.20                           # [TB]
KD = 0.15                           # [TB]
DEADBAND = 0.08                     # [TB] |u| menor que esto → recto

# ─────────────────────────────────────────────────────────────────────────────
#  RECUPERACIÓN (línea perdida)
# ─────────────────────────────────────────────────────────────────────────────
LOST_GRACE_S = 0.25                 # Pérdidas más cortas se ignoran (parpadeo)
RECOVERY_U = 0.9                    # Intensidad de giro hacia el último lado visto
LOST_STOP_S = 8.0                   # Sin línea tanto tiempo → parar por seguridad

# ─────────────────────────────────────────────────────────────────────────────
#  ROBOT  ← lo que se ajusta cuando el profesor entregue el robot
# ─────────────────────────────────────────────────────────────────────────────
# Conexión:
#   None                 → simulación (no se conecta nada, se ve en "Movimiento")
#   "00:1B:10:21:2C:1B"  → MAC del mBot (socket RFCOMM, como master_pc/Robot.py)
#   "/dev/rfcomm0"/"COM5"→ puerto serie (pyserial)
ROBOT_ADDRESS = None
RFCOMM_CHANNEL = 1
BAUD_RATE = 115200

# Cómo se le habla al robot (ver drivers.py):
#   "pulsos"      → firmware actual del curso: letras w/a/d/x, pulsos fijos
#   "velocidades" → firmware que acepte velocidad por rueda, p. ej. "V:120,80\n"
DRIVER = "pulsos"

# --- Driver "pulsos" (arduinoFinal.ino) ---
PULSE_FORWARD_S = 0.100             # Lo que dura 'w' en el firmware
PULSE_TURN_S = 0.030                # Lo que dura 'a'/'d' en el firmware
LINK_MARGIN_S = 0.010               # Holgura por latencia del Bluetooth
MAX_TURN_PULSES = 4                 # [TB] Pulsos de giro por cada 'w' cuando |u| = 1
SPIN_THRESHOLD = 0.85               # |u| mayor → gira sin avanzar (curva cerrada)
PULSE_COMMANDS = {                  # Por si el firmware entregado usa otras letras
    "forward": "w", "left": "a", "right": "d", "stop": "x",
}

# --- Driver "velocidades" ---
BASE_SPEED = 150                    # [TB] 0-255
MAX_SPEED = 255
SPEED_FORMAT = "V:{left},{right}\n"  # Plantilla del mensaje
SPEED_PERIOD_S = 0.05               # Un mensaje cada 50 ms (20 Hz)

# --- Modelo aproximado del robot (solo para la ventana "Movimiento") ---
SIM_FORWARD_CM_PER_S = 20.0         # Velocidad lineal con 'w' / BASE_SPEED
SIM_TURN_DEG_PER_S = 180.0          # Velocidad de giro con 'a'/'d'
SIM_WHEEL_BASE_CM = 11.5            # Distancia entre ruedas del mBot

# ─────────────────────────────────────────────────────────────────────────────
#  SEÑALES (objetivos 5 y 6 del reto)
# ─────────────────────────────────────────────────────────────────────────────
# Rangos HSV [TB] ajustables con trackbars.
# Rojo: dos rangos de matiz por el-wrap en HSV (0-10 y 170-180).
# Verde: un rango continuo.
RED_HUE1_LOW, RED_HUE1_HIGH = 0, 10
RED_HUE2_LOW, RED_HUE2_HIGH = 170, 180
GREEN_HUE_LOW, GREEN_HUE_HIGH = 45, 90
SIGNAL_MIN_SAT = 60          # S mínima para considerar un color "saturo"
SIGNAL_MIN_VAL = 60          # V mínima
MIN_SIGNAL_AREA = 0.005      # Área mínima de la señal (fracción del frame total)
SIGNAL_CONFIRM_FRAMES = 3    # N° de frames consecutivos para confirmar detección
STOP_DURATION_S = 3.0        # Tiempo que el robot se detiene ante PARE (docente/trackbar)
SIGNAL_COOLDOWN_S = 2.0      # Ignorar la misma señal X s después de actuar
SIGNAL_COVER_GRACE_S = 0.8   # Si la línea desaparece justo después de ver señal,
                             # se asume que está tapada → seguir recto X s
EPSILON_POLYDP_RATIO = 0.03  # Ratio del perímetro para approxPolyDP (7-9 vértices)
# Cámara virtual: trapecio de piso que ve, medido desde el centro del robot.
SIM_CAM_NEAR_CM = 8.0               # Distancia al borde inferior de la imagen
SIM_CAM_DEPTH_CM = 25.0             # Profundidad del campo de visión
SIM_CAM_NEAR_WIDTH_CM = 16.0        # Ancho visto en el borde inferior
SIM_CAM_FAR_WIDTH_CM = 32.0         # Ancho visto en el borde superior (perspectiva)
SIM_CAM_HEIGHT_PX = 240             # Alto del frame virtual (ancho = PROCESS_WIDTH)
SIM_CAM_FPS = 20.0                  # Frames por segundo de la cámara virtual
SIM_LATENCY_S = 0.10                # Retraso cámara→decisión (WiFi + procesamiento)
