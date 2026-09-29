"""
Detección de señales de tráfico: rombo/hexágono/octágono rojo (PARE) y rombo/octágono verde (SIGA).

Operaciones permitidas (visión clásica, sin ML):
- Conversión HSV, inRange, morfología (apertura/cierre).
- findContours, approxPolyDP para contar vértices.
- Filtros de área, solidez y relación de aspecto.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (RED_HUE1_LOW, RED_HUE1_HIGH, RED_HUE2_LOW, RED_HUE2_HIGH,
                    GREEN_HUE_LOW, GREEN_HUE_HIGH, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL,
                    MIN_SIGNAL_AREA, SIGNAL_MIN_MEAN_SAT, SIGNAL_MIN_SOLIDITY, SIGNAL_ASPECT_RANGE,
                    PARE_VERTICES, SIGA_VERTICES, EPSILON_POLYDP_RATIO)

# Rango de vértices aceptado por cada señal (después de approxPolyDP)
_VERTICES = {"PARE": PARE_VERTICES, "SIGA": SIGA_VERTICES}
_KERNEL5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


@dataclass
class SignalResult:
    """Señal encontrada por detect_signal(): qué es, dónde está y su forma."""
    label: str              # "PARE" o "SIGA"
    center: Tuple[float, float]  # centroide (x, y) en píxeles del frame
    area: float             # área del contorno en píxeles
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box
    vertices: int           # número de vértices después de approxPolyDP
    contour: np.ndarray   # polígono aproximado (coordenadas del frame)
    partial: bool = False   # Cortada por el borde: no se usa para actuar, solo para
                            # saber que hay una señal (y una barra) por delante


def detect_signal(frame: np.ndarray) -> Optional[SignalResult]:
    """
    Busca una señal de tráfico (polígono rojo = PARE, polígono verde = SIGA) en el frame.

    La imagen se convierte a HSV antes de la segmentación por color.
    Escanea el frame completo (la señal está tendida sobre la línea, en el piso).
    Retorna el mejor resultado encontrado, o None si no hay señal confiable.

    Pasos: máscara de color → limpieza morfológica → contornos → filtros
    (área mínima, solidez, aspecto, n° de vértices) → se queda con el de mayor área.
    """

    # Convertir a espacio HSV (el canal V ayuda a distinguir objetos coloreados
    # de sombras, igual al diseño de vision.py para la línea negra).
    # CRÍTICO: el frame debe venir en BGR (3 canales); si no, cvtColor falla.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]
    best: Optional[SignalResult] = None
    partial: Optional[SignalResult] = None
    best_area = 0.0

    # Probar ambas etiquetas: PARE (rojo) y SIGA (verde)
    for label in ("PARE", "SIGA"):
        # Construir máscara HSV
        # CRÍTICO: los límites se pasan como np.uint8 (mismo tipo que la imagen);
        # con otro tipo cv2.inRange puede lanzar error.
        if label == "PARE":
            # Rojo: dos rangos de matiz por el-wrap en HSV (0-10 y 170-180)
            mask1 = cv2.inRange(hsv,
                                np.array([RED_HUE1_LOW, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL], dtype=np.uint8),
                                np.array([RED_HUE1_HIGH, 255, 255], dtype=np.uint8))
            mask2 = cv2.inRange(hsv,
                                np.array([RED_HUE2_LOW, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL], dtype=np.uint8),
                                np.array([RED_HUE2_HIGH, 255, 255], dtype=np.uint8))
            mask = cv2.bitwise_or(mask1, mask2)     # Unir los dos rangos de rojo
        else:  # SIGA
            # Verde: un rango continuo de matiz
            mask = cv2.inRange(hsv,
                               np.array([GREEN_HUE_LOW, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL], dtype=np.uint8),
                               np.array([GREEN_HUE_HIGH, 255, 255], dtype=np.uint8))

        # Morfología: apertura (quita ruido), cierre (rellena huecos pequeños)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL5)

        # Encontrar contornos (solo los externos), del más grande al más pequeño
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            area = cv2.contourArea(c)
            # Área mínima relativa al tamaño del frame (descarta manchas pequeñas)
            if area < MIN_SIGNAL_AREA * w * h:
                continue

            # Filtro de solidez (area / area del convex hull)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            # CRÍTICO: evita ZeroDivisionError en la división de la solidez
            if hull_area == 0:
                continue
            solidity = area / hull_area
            # Un polígono regular es convexo (solidez ≈ 1); la madera o una
            # mancha irregular rojiza queda por debajo
            if solidity < SIGNAL_MIN_SOLIDITY:
                continue

            # Saturación promedio dentro del contorno (cv2.mean con máscara =
            # Σ S / n° de píxeles). Una señal pintada tiene color "puro"; la
            # madera o la piel pasan el umbral por píxel en algunos puntos
            # pero en promedio son apagadas.
            cmask = np.zeros((h, w), np.uint8)
            cv2.drawContours(cmask, [c], -1, 255, cv2.FILLED)
            if cv2.mean(hsv, mask=cmask)[1] < SIGNAL_MIN_MEAN_SAT:
                continue

            # Relación de aspecto ancho/alto del bounding box: un hexágono u
            # octágono regular ≈ 1; una tabla o una franja alargada no
            x, y, bw, bh = cv2.boundingRect(c)
            # Si toca el borde del frame solo se ve un trozo: su forma no es la
            # real (una tira de papel cortada puede parecer un rombo). No sirve
            # para actuar; se devuelve como "parcial" si no hay otra mejor.
            if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
                if partial is None:
                    partial = SignalResult(label, (x + bw / 2.0, y + bh / 2.0), area,
                                           (x, y, bw, bh), 0, c, partial=True)
                continue
            aspect = bw / float(bh)
            if not (SIGNAL_ASPECT_RANGE[0] <= aspect <= SIGNAL_ASPECT_RANGE[1]):
                continue

            # approxPolyDP para contar vértices
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            # epsilon = 3 % del perímetro: distancia máxima entre el contorno
            # real y el polígono simplificado (más grande → menos vértices)
            epsilon = EPSILON_POLYDP_RATIO * perimeter
            approx = cv2.approxPolyDP(c, epsilon, True)
            vertices = len(approx)

            # PARE: rombo (4), hexágono (6) u octágono (8); SIGA: rombo u octágono.
            # La tolerancia cubre la perspectiva y los bordes imperfectos.
            vmin, vmax = _VERTICES[label]
            if not (vmin <= vertices <= vmax):
                continue

            # Centro del bounding box (coordenadas del frame, para dibujar)
            cx = x + bw / 2.0
            cy = y + bh / 2.0

            # Quedarse con la señal más grande (la más cercana a la cámara)
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

    return best if best is not None else partial
