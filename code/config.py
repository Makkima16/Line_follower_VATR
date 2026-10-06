"""
Parámetros del seguidor de línea.

Cuando llegue el robot, normalmente solo hay que tocar la sección
"ROBOT" de este archivo. Los marcados con [TB] también se ajustan en vivo
desde la ventana de trackbars "Calibracion".
"""

# ─────────────────────────────────────────────────────────────────────────────
#  CÁMARA
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_SOURCE = "phone"               # "phone" (celular por WiFi) o "usb"
CAMERA_INDEX = 0                    # Solo para "usb"
PHONE_HTTP_PORT = 9999
PHONE_HTTPS_PORT = 8433

# Según cómo quede montado el celular en el robot:
ROTATE = 0                          # 0, 90, 180 o 270 grados (sentido horario)
FLIP_HORIZONTAL = False             # Espejo izquierda/derecha

PROCESS_WIDTH = 320                 # Se reescala a este ancho (se conserva la proporción)
DISPLAY_SCALE = 2.0                 # Ampliación de la ventana de depuración

# ─────────────────────────────────────────────────────────────────────────────
#  DETECCIÓN DE LA LÍNEA (negra sobre fondo claro)
# ─────────────────────────────────────────────────────────────────────────────
ROI_HEIGHT_RATIO = 0.60             # [TB] Se analiza este % inferior del frame
CHASSIS_MASK_RATIO = 0.34          # [TB] Fracción inferior del frame que es el propio robot             # [TB] Se analiza este % inferior del frame
N_SLICES = 5                        # Franjas horizontales dentro de la ROI
MIN_BLOB_AREA_RATIO = 0.01          # Área mínima de un trozo de línea (fracción de la franja)
MAX_JUMP_RATIO = 0.25               # Salto horizontal máx. entre franjas (fracción del ancho)
NEAR_WEIGHT = 0.6                   # [TB] Peso del punto cercano frente al lejano en el error
BINARY_THRESHOLD = 80              # [TB] 0 = Otsu automático

# ─────────────────────────────────────────────────────────────────────────────
#  CONTROL PD
# ─────────────────────────────────────────────────────────────────────────────
# El error se normaliza a [-1, 1] (±1 = línea en el borde de la imagen) y la
# salida u también: u = +1 "gira todo a la derecha", u = -1 "todo a la
# izquierda". Así el control no depende ni de la cámara ni del robot.
KP = 0.50                           # [TB]
KD = 0.05                           # [TB]
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
ROBOT_ADDRESS = "00:1B:10:21:2C:1B"
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
ERROR_SMOOTHING = 0.7                # Peso del EMA del error (0.7 = lento, 0 = sin filtro)
D_SMOOTHING = 0.8                 # Peso del filtrado derivativo (0.8 = lento, 0.5 = original)
FORWARD_DUTY = 0.45                 # [TB] Fracción de ciclos que avanza (0=quieto, 1=sin acelerador)
SPIN_THRESHOLD = 0.85               # |u| mayor → gira sin avanzar (curva cerrada)
PULSE_COMMANDS = {                  # Por si el firmware entregado usa otras letras
    "forward": "w", "left": "a", "right": "d", "stop": "x",
}

# --- Driver "velocidades" ---
BASE_SPEED = 50                    # [TB] 0-255
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
SIGNAL_MIN_SAT = 60          # S mínima por píxel para entrar en la máscara
SIGNAL_MIN_MEAN_SAT = 90     # S promedio dentro del contorno: señal pintada ≈ 150, madera/piel ≈ 40
SIGNAL_MIN_VAL = 60          # V mínima
SIGNAL_BAND = (0.30, 0.98)       # Fracción vertical del frame donde puede haber señal
MIN_SIGNAL_AREA = 0.003      # Área mínima de la señal (fracción del frame total)
SIGNAL_MIN_SOLIDITY = 0.80   # área/área_casco: polígonos regulares ≈ 0.95; descarta madera, sombras
SIGNAL_ASPECT_RANGE = (0.6, 1.6)   # ancho/alto del bounding box (un polígono regular ≈ 1)
PARE_VERTICES = (4, 9)       # Rojo: rombo/cuadrado (4), hexágono (6) u octágono (8), con tolerancia
SIGA_VERTICES = (4, 9)       # Verde: rombo/cuadrado (4) u octágono (8)
# Con 4 vértices la relación de aspecto (SIGNAL_ASPECT_RANGE) y la solidez son las
# que descartan tiras de papel o rectángulos alargados del mismo color.
SIGNAL_CONFIRM_FRAMES = 3    # N° de frames consecutivos para confirmar detección
STOP_DURATION_S = 3.0        # [TB] Tiempo que el robot se detiene ante PARE
SIGNAL_COOLDOWN_S = 1.0      # La señal debe DESAPARECER este tiempo antes de volver a actuar
                             # (si no, al reanudar la vería otra vez y pararía para siempre)
SIGNAL_COVER_GRACE_S = 0.8   # Si la línea desaparece justo después de ver señal,
                             # se asume que está tapada → seguir recto X s
EPSILON_POLYDP_RATIO = 0.03  # epsilon de approxPolyDP como fracción del perímetro

# ─────────────────────────────────────────────────────────────────────────────
#  RAMIFICACIONES Y MAPA (mapa.py)
# ─────────────────────────────────────────────────────────────────────────────
# Salidas de la línea por el borde del FRAME completo (izq., arriba, der.). Solo es
# camino lo que sale de la imagen; un brazo que termina a la vista (barra transversal)
# no cuenta:
#   1 salida  → tramo normal      2+ salidas → ramificación
#   0 salidas → la línea termina a la vista → posible callejón sin salida
MIN_EXIT_PX = 4                     # Largo mínimo de un tramo blanco en el borde para contar como salida
EXIT_MERGE_RATIO = 1.0              # Tramos separados por menos de este × ancho de la línea son UNA salida
                                    # (un brillo en la cinta hace una muesca y parte la salida en dos)
ENTRY_SIDE_RATIO = 0.20             # Lo que toca los lados en este % inferior del frame es la entrada, no una salida
JUNCTION_CONFIRM_FRAMES = 2         # Frames seguidos con 2+ salidas para aceptar la ramificación
CROSSBAR_WIDTH_RATIO = 2.5          # Fila de la mancha así de ancha (× ancho de línea) y cerrada = barra transversal
CROSSBAR_IGNORE_CM = 70.0           # Tras ver una barra entera, no aceptar cruces estos cm (hasta pasarla)
DEAD_END_FRAMES = 5                 # Frames seguidos con 0 salidas para aceptar el callejón
DEAD_END_REACH = 0.70               # ...y la línea no pasa de este % de la altura del frame
                                    # (se ve piso vacío más allá del final: no hay por dónde seguir)
BRANCH_POLICY = "izquierda"         # Orden de exploración: "izquierda", "derecha" o "recto"
BRANCH_MATCH_DEG = 50.0             # Tolerancia para reconocer una rama ya vista por su rumbo
NODE_MATCH_CM = 30.0                # Radio para reconocer un cruce ya visitado por odometría
COMMIT_CM = 35.0                    # Tras decidir en un cruce, se mantiene la rama elegida estos cm
COMMIT_MAX_FRAMES = 120             # ...o como máximo estos frames (≈6 s): nunca quedarse trabado
HEADING_FIX_GAIN = 0.7              # Al volver a un cruce, cuánto se corrige el rumbo de la odometría (0-1)
NODE_REACH_CM = 4.0                 # Si la rama elegida no se ve, avanzar hasta el cruce y girar ahí
STEER_FULL_DEG = 40.0               # Ángulo hacia la rama que equivale a error = ±1
UTURN_U = 1.0                       # Mando de giro durante la media vuelta (giro en sitio)
UTURN_MIN_DEG = 140.0               # No aceptar la línea hasta haber girado esto (sigue viendo el callejón)
UTURN_MAX_DEG = 400.0               # Más de una vuelta sin encontrar línea → BUSCANDO

# ─────────────────────────────────────────────────────────────────────────────
#  CÁMARA SOBRE EL PISO
# ─────────────────────────────────────────────────────────────────────────────
# Trapecio de piso que ve la cámara, medido desde el centro del robot.
# Lo usan el simulador (cámara virtual) Y el mapa (pasar píxeles a cm con una
# homografía). En el robot real: poner una hoja en el piso y medir estas 4 cotas.
SIM_CAM_NEAR_CM = 8.0               # Distancia al borde inferior de la imagen
SIM_CAM_DEPTH_CM = 25.0             # Profundidad del campo de visión
SIM_CAM_NEAR_WIDTH_CM = 16.0        # Ancho visto en el borde inferior
SIM_CAM_FAR_WIDTH_CM = 32.0         # Ancho visto en el borde superior (perspectiva)
SIM_CAM_HEIGHT_PX = 240             # Alto del frame virtual (ancho = PROCESS_WIDTH)
SIM_MAP_WIDTH_CM = 120.0             # Ancho del mapa del simulador en cm
SIM_CAM_FPS = 20.0                  # Frames por segundo de la cámara virtual
SIM_LATENCY_S = 0.10                # Retraso cámara→decisión (WiFi + procesamiento)
