"""
Detección de señales de tráfico: octágonos rojo (PARE) y verde (SIGA).

Operaciones permitidas (visión clásica, sin ML):
- Conversión HSV, inRange, morfología (apertura/cierre).
- findContours, approxPolyDP para contar vértices.
- Filters de área, solidez y ratio de perímetro.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (RED_HUE1_LOW, RED_HUE1_HIGH, RED_HUE2_LOW, RED_HUE2_HIGH,
                    GREEN_HUE_LOW, GREEN_HUE_HIGH, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL,
                    MIN_SIGNAL_AREA)


@dataclass
class SignalResult:
    label: str              # "PARE" o "SIGA"
    center: Tuple[float, float]  # centroide (x, y) en píxeles del frame
    area: float             # área del contorno en píxeles
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box
    vertices: int           # número de vértices después de approxPolyDP
    contour: np.ndarray   # contorno del octágono (coordenadas del frame)


def detect_signal(frame: np.ndarray) -> Optional[SignalResult]:
    """
    Busca una señal de tráfico (octágono rojo o verde) en el frame.

    La imagen se convierte a HSV antes de la segmentación por color.
    Escanea el frame completo (la señal está tendida sobre la línea, en el piso).
    Retorna el mejor resultado encontrado, o None si no hay señal confiable.
    """

    # Convertir a espacio HSV (el canal V ayuda a distinguir objetos coloreados
    # de sombras, igual al diseño de vision.py para la línea negra).
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]
    best: Optional[SignalResult] = None
    best_area = 0.0

    # Probar ambas etiquetas: PARE (rojo) y SIGA (verde)
    for label in ("PARE", "SIGA"):
        # Construir máscara HSV
        if label == "PARE":
            # Rojo: dos rangos de matiz por el-wrap en HSV (0-10 y 170-180)
            mask1 = cv2.inRange(hsv,
                                np.array([RED_HUE1_LOW, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL], dtype=np.uint8),
                                np.array([RED_HUE1_HIGH, 255, 255], dtype=np.uint8))
            mask2 = cv2.inRange(hsv,
                                np.array([RED_HUE2_LOW, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL], dtype=np.uint8),
                                np.array([RED_HUE2_HIGH, 255, 255], dtype=np.uint8))
            mask = cv2.bitwise_or(mask1, mask2)
        else:  # SIGA
            # Verde: un rango continuo de matiz
            mask = cv2.inRange(hsv,
                               np.array([GREEN_HUE_LOW, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL], dtype=np.uint8),
                               np.array([GREEN_HUE_HIGH, 255, 255], dtype=np.uint8))

        # Morfología: apertura (quita ruido), cierre (rellena huecos pequeños)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # Encontrar contornos
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < MIN_SIGNAL_AREA * w * h:
                continue

            # Filtro de solidez (area / area del convex hull)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < 0.5:          # octágonos convexos; descartar formas raras
                continue

            # approxPolyDP para contar vértices
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            epsilon = 0.03 * perimeter
            approx = cv2.approxPolyDP(c, epsilon, True)
            vertices = len(approx)

            # Queremos aproximadamente 8 vértices (tolerancia 7-9 por deformación perspectiva)
            if not (7 <= vertices <= 9):
                continue

            # Bounding box y centroide (sobre la imagen original para dibujo)
            # Usamos el frame original (no HSV) para obtener coordenadas consistentes
            x, y, bw, bh = cv2.boundingRect(c)
            cx = x + bw / 2.0
            cy = y + bh / 2.0

            if area > best_area:
                best_area = area
                best = SignalResult(
                    label=label,
                    center=(cx, cy),
                    area=area,
                    bbox=(x, y, bw, bh),
                    vertices=vertices,
                    contour=approx,
                )

    return best