"""
Detección de señales con VGG16 (rama vgg16_focus).

Misma interfaz que senales.detect_signal(), para poder intercambiarlas desde
config.SIGNAL_DETECTOR.

VGG16 recibe el FRAME COMPLETO, igual que las fotos con las que se entrenó
(cuadro entero con la señal pequeña sobre la línea) y decide entre 5 clases:
pare, siga, velocidad, zona escolar y fondo (pista sin señal). Solo PARE y SIGA
mueven la máquina de estados del robot (control.Brain).

Para no bajar los FPS, la red se ejecuta cada config.VGG16_CADA_N_FRAMES frames;
en los demás se repite el último resultado. La confirmación de varios frames
consecutivos la hace Brain (config.SIGNAL_CONFIRM_FRAMES).
"""

import os
from typing import Optional

import cv2
import numpy as np

import config
from senales import SignalResult   # Mismo contenedor de datos que la versión clásica

# Orden de las carpetas del dataset (0, 1, 2, 3, 4), igual que en predict.py
CLASES = ["pare", "siga", "velocidad", "zona escolar", "fondo"]
ETIQUETA_ROBOT = {"pare": "PARE", "siga": "SIGA"}   # Las únicas que mueven la máquina de estados

IMG_SIZE = (224, 224)
RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "results", config.VGG16_EXPERIMENTO, "VGG16", "VGG16.keras")

_modelo = None          # Se carga la primera vez que se necesita (TensorFlow tarda en importar)
_contador = 0           # Frames vistos, para ejecutar la red solo cada N
_ultimo = None          # Último resultado de la red, que se repite entre ejecuciones


def _cargar_modelo():
    global _modelo
    if _modelo is None:
        import keras   # Import tardío: no frena el arranque si se usa el detector clásico
        ruta = os.path.normpath(RUTA_MODELO)
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No existe el modelo VGG16: {ruta}. Entrénalo con main.py de la raíz.")
        _modelo = keras.models.load_model(ruta)
        print(f"[VGG16] Modelo cargado: {ruta}")
    return _modelo


def clasificar(frame_bgr: np.ndarray):
    """Devuelve (clase, confianza) para un frame BGR. Mismo preprocesado que en el entrenamiento."""
    from keras import applications
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    x = applications.vgg16.preprocess_input(np.expand_dims(rgb, axis=0))
    probs = _cargar_modelo().predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASES[idx], float(probs[idx])


def detect_signal(frame: np.ndarray,
                  min_frac: float = 0.0,
                  max_frac: float = 1.0) -> Optional[SignalResult]:
    """Clasifica el frame completo; devuelve SignalResult si es PARE o SIGA con confianza suficiente.

    min_frac y max_frac se aceptan por compatibilidad con senales.detect_signal pero se
    ignoran: la red necesita ver el cuadro entero, como en el entrenamiento. Como VGG16
    no localiza la señal, center y bbox describen el frame completo.
    """
    global _contador, _ultimo
    if _contador % config.VGG16_CADA_N_FRAMES == 0:
        clase, conf = clasificar(frame)
        _ultimo = (clase, conf) if conf >= config.VGG16_MIN_CONF else None
    _contador += 1

    if _ultimo is None or _ultimo[0] not in ETIQUETA_ROBOT:
        return None

    h, w = frame.shape[:2]
    caja = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], np.int32)
    return SignalResult(label=ETIQUETA_ROBOT[_ultimo[0]], center=(w / 2.0, h / 2.0), area=float(w * h),
                        bbox=(0, 0, w, h), vertices=0, contour=caja)
