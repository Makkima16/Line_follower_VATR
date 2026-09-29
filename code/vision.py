"""
Visión clásica de la línea: umbral, morfología, contornos y momentos.
"""

from dataclasses import dataclass

import cv2
import numpy as np

import config

_KERNEL5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


@dataclass
class LineResult:
    points: list          # Centroide por franja, de abajo (cerca) hacia arriba (lejos)
    contours: list        # Contorno elegido en cada franja (coords. del frame)
    binary: np.ndarray    # ROI binarizada (línea = blanco)
    roi_top: int
    error: float          # Error normalizado en [-1, 1]; > 0 = línea a la derecha


def prepare_frame(frame):
    """Orienta el frame según el montaje de la cámara y lo reescala."""
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
           270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(config.ROTATE)
    if rot is not None:
        frame = cv2.rotate(frame, rot)
    if config.FLIP_HORIZONTAL:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if w != config.PROCESS_WIDTH:
        new_h = int(round(h * config.PROCESS_WIDTH / w))
        frame = cv2.resize(frame, (config.PROCESS_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    return frame


def line_mask(roi, threshold):
    """
    Máscara de la línea negra (línea = 255).

    Se umbraliza V = max(R, G, B) (canal V de HSV) en lugar del gris: así
    solo cuenta como "negro" lo que es oscuro en los tres canales, y
    cualquier objeto de color vivo (V alto) no se confunde con la línea.
    """
    v = cv2.GaussianBlur(roi.max(axis=2), (5, 5), 0)
    if threshold <= 0:
        _, binary = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(v, threshold, 255, cv2.THRESH_BINARY_INV)
    # Cierre: rellena brillos dentro de la cinta. Apertura: borra puntos sueltos.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _KERNEL5)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, _KERNEL5)


def detect_line(frame, threshold, roi_ratio=None, near_weight=None, prev_x=None):
    """
    Busca la línea por franjas horizontales dentro de la ROI inferior.

    De abajo hacia arriba, en cada franja se toma el trozo (contorno) cuyo
    centroide x = m10/m00 está más cerca del de la franja anterior. Así se
    sigue la línea en curvas y se ignoran manchas sueltas. El error mezcla
    el punto cercano (posición actual) con el lejano (hacia dónde va la
    curva) para empezar a girar antes de llegar a ella.
    """
    roi_ratio = config.ROI_HEIGHT_RATIO if roi_ratio is None else roi_ratio
    near_weight = config.NEAR_WEIGHT if near_weight is None else near_weight

    h, w = frame.shape[:2]
    roi_top = int(h * (1.0 - roi_ratio))
    binary = line_mask(frame[roi_top:, :], threshold)

    roi_h = h - roi_top
    slice_h = max(roi_h // config.N_SLICES, 1)
    min_area = config.MIN_BLOB_AREA_RATIO * slice_h * w
    max_jump = config.MAX_JUMP_RATIO * w

    ref_x = prev_x if prev_x is not None else w / 2.0
    points, contours = [], []

    for i in range(config.N_SLICES):
        y1 = roi_h - i * slice_h
        y0 = 0 if i == config.N_SLICES - 1 else y1 - slice_h
        cnts, _ = cv2.findContours(binary[y0:y1, :], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            m = cv2.moments(c)
            if m["m00"] < min_area:
                continue
            x = m["m10"] / m["m00"]
            dist = abs(x - ref_x)
            if best is None or dist < best[0]:
                best = (dist, x, m["m01"] / m["m00"], c)

        if best is None:
            if points:
                break          # La línea termina o sale del cuadro
            continue           # Franja de abajo vacía: probar la siguiente
        dist, x, y, c = best
        if points and dist > max_jump:
            break              # Salto brusco: es otra mancha

        c = c.copy()
        c[:, :, 1] += roi_top + y0
        points.append((x, roi_top + y0 + y))
        contours.append(c)
        ref_x = x

    if not points:
        return None

    center = w / 2.0
    near_x, far_x = points[0][0], points[-1][0]
    error_px = near_weight * (near_x - center) + (1.0 - near_weight) * (far_x - center)
    error = float(np.clip(error_px / center, -1.0, 1.0))
    return LineResult(points, contours, binary, roi_top, error)
