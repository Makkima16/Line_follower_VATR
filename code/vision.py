"""
Visión clásica de la línea: umbral, morfología, contornos y momentos.

Flujo: prepare_frame() orienta/reescala la imagen → detect_line() recorta la
parte de abajo (ROI), la binariza con line_mask() y calcula el centroide de
la línea en varias franjas para obtener el error que usa el control.

Además find_exits() recorre el borde del FRAME COMPLETO y cuenta por dónde
"sale" la línea: 1 salida = tramo normal, 2 o más = ramificación, 0 = la
línea termina a la vista (callejón). Un brazo que termina dentro de la imagen
(p. ej. una barra transversal) no es camino: se ve que no lleva a ningún lado. pixel_to_floor() pasa esos puntos a cm sobre el piso
para que el mapa (mapa.py) sepa hacia qué rumbo va cada rama.
"""

import math

from dataclasses import dataclass

import cv2
import numpy as np

import config

# Kernel 5x5 para abrir/cerrar la máscara. Se crea una sola vez (no en cada frame).
_KERNEL5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


@dataclass
class Exit:
    """Un punto por donde la línea sale del frame (borde izquierdo, superior o derecho)."""
    px: tuple             # (x, y) en píxeles del frame
    floor: tuple          # (adelante, derecha) en cm desde el centro del robot
    angle: float          # Ángulo desde el robot hasta la salida, en grados (> 0 = derecha)


@dataclass
class LineResult:
    """Resultado de detect_line(): todo lo que se sabe de la línea en este frame."""
    points: list          # Centroide por franja, de abajo (cerca) hacia arriba (lejos)
    contours: list        # Contorno elegido en cada franja (coords. del frame)
    binary: np.ndarray    # ROI binarizada (línea = blanco)
    roi_top: int          # Fila del frame donde empieza la ROI
    error: float          # Error normalizado en [-1, 1]; > 0 = línea a la derecha
    exits: list           # Salidas (Exit) de la línea por el borde del frame
    reach: float          # Hasta dónde llega la línea: 1 = toca el borde superior del frame
    others: int           # Otros trozos de línea en el frame (p. ej. después de un hueco en la cinta)
    crossbar: bool        # Se ve una barra transversal con sus DOS extremos dentro de la imagen
    entry_floor: tuple    # Punto cercano de la línea en cm (adelante, derecha)


def prepare_frame(frame):
    """
    Orienta el frame según el montaje de la cámara y lo reescala.

    Reducir el ancho a config.PROCESS_WIDTH hace que todo el procesamiento
    sea más rápido y que los parámetros en píxeles valgan igual con
    cualquier cámara.
    """
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
           270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(config.ROTATE)
    if rot is not None:
        frame = cv2.rotate(frame, rot)
    if config.FLIP_HORIZONTAL:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if w != config.PROCESS_WIDTH:
        # Se conserva la proporción: nuevo_alto = alto · (ancho_nuevo / ancho)
        new_h = int(round(h * config.PROCESS_WIDTH / w))
        frame = cv2.resize(frame, (config.PROCESS_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    return frame


def line_mask(roi, threshold):
    """
    Máscara de la línea negra (línea = 255).

    Se umbraliza V = max(R, G, B) (canal V de HSV) en lugar del gris: así
    solo cuenta como "negro" lo que es oscuro en los tres canales, y
    cualquier objeto de color vivo (V alto) no se confunde con la línea.

    threshold = 0 → Otsu elige el umbral solo (separa los dos picos del
    histograma); threshold > 0 → umbral fijo del trackbar.
    """
    # CRÍTICO: roi debe ser una imagen a color (3 canales); max(axis=2) toma el
    # mayor de B, G, R por píxel. Con una imagen en gris esta línea falla.
    v = cv2.GaussianBlur(roi.max(axis=2), (5, 5), 0)
    if threshold <= 0:
        _, binary = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # BINARY_INV: lo oscuro (la línea) queda en blanco = 255
        _, binary = cv2.threshold(v, threshold, 255, cv2.THRESH_BINARY_INV)
    # Cierre: rellena brillos dentro de la cinta. Apertura: borra puntos sueltos.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _KERNEL5)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, _KERNEL5)


def detect_line(frame, threshold, roi_ratio=None, near_weight=None, prev_x=None,
                chassis_ratio=None):
    """
    Busca la línea por franjas horizontales dentro de la ROI inferior.

    De abajo hacia arriba, en cada franja se toma el trozo (contorno) cuyo
    centroide x = m10/m00 está más cerca del de la franja anterior. Así se
    sigue la línea en curvas y se ignoran manchas sueltas. El error mezcla
    el punto cercano (posición actual) con el lejano (hacia dónde va la
    curva) para empezar a girar antes de llegar a ella.

    Parámetros:
      threshold   : umbral de binarización (0 = Otsu).
      roi_ratio   : fracción inferior del frame que se analiza (0..1).
      near_weight : peso del punto cercano en el error (0..1).
      prev_x      : x de la línea en el frame anterior (para no saltar a otra mancha).

    Devuelve un LineResult, o None si no se encontró la línea.
    """
    roi_ratio = config.ROI_HEIGHT_RATIO if roi_ratio is None else roi_ratio
    near_weight = config.NEAR_WEIGHT if near_weight is None else near_weight
    chassis_ratio = (config.CHASSIS_MASK_RATIO if chassis_ratio is None
                     else chassis_ratio)

    h, w = frame.shape[:2]
    roi_top = int(h * (1.0 - roi_ratio))
    # Se binariza el frame entero: la ROI se usa para el error (lo cercano) y el
    # frame completo para ver hasta dónde llega la línea (salidas / callejón)
    full = line_mask(frame, threshold)
    # El cuerpo del robot ocupa la franja inferior y es oscuro y continuo, asi que
    # la binarizacion lo marca como si fuera pista. Como ademas es mas ancho y
    # limpio que la linea real, compite con ella y la franja mas cercana (la de
    # mayor peso en el error) cae sobre el chasis. Se pone en blanco antes de
    # cortar las franjas para que el detector no lo vea.
    # La banda de analisis va desde roi_top hasta justo encima del chasis. Si se
    # dejara hasta el fondo, las franjas bajas se wasting en filas ya en blanco y
    # ademas la franja mas cercana (la de mayor peso en el error) no seria la
    # correcta: el error se mediria contra el borde del robot en vez de la pista.
    cut = int(h * chassis_ratio)
    band_bottom = max(h - cut, roi_top)
    if cut > 0:
        full[band_bottom:, :] = 255
    binary = full[roi_top:band_bottom, :]

    roi_h = band_bottom - roi_top
    if roi_h < config.N_SLICES:      # ROI inutilizable: usar todo lo que haya
        band_bottom = h - cut if cut > 0 else h
        binary = full[roi_top:band_bottom, :]
        roi_h = band_bottom - roi_top
    # CRÍTICO: max(..., 1) evita franjas de 0 px de alto si la ROI es muy baja
    # (con 0 las franjas quedarían vacías y nunca se encontraría la línea).
    slice_h = max(roi_h // config.N_SLICES, 1)
    min_area = config.MIN_BLOB_AREA_RATIO * slice_h * w   # Área mínima de un trozo válido
    max_jump = config.MAX_JUMP_RATIO * w                  # Salto lateral máximo entre franjas

    # Referencia inicial: donde estaba la línea antes, o el centro de la imagen
    ref_x = prev_x if prev_x is not None else w / 2.0
    points, contours = [], []

    for i in range(config.N_SLICES):
        # Franja i contada desde abajo; la última se estira hasta arriba de la ROI
        y1 = roi_h - i * slice_h
        y0 = 0 if i == config.N_SLICES - 1 else y1 - slice_h
        cnts, _ = cv2.findContours(binary[y0:y1, :], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            m = cv2.moments(c)
            # CRÍTICO: este filtro también descarta m00 = 0; sin él la división
            # m10/m00 de abajo lanzaría ZeroDivisionError.
            if m["m00"] < min_area:
                continue
            x = m["m10"] / m["m00"]          # Centroide x = m10 / m00
            dist = abs(x - ref_x)
            if best is None or dist < best[0]:
                best = (dist, x, m["m01"] / m["m00"], c)   # Centroide y = m01 / m00

        if best is None:
            if points:
                break          # La línea termina o sale del cuadro
            continue           # Franja de abajo vacía: probar la siguiente
        dist, x, y, c = best
        if points and dist > max_jump:
            break              # Salto brusco: es otra mancha

        # Pasar el contorno de coordenadas de la franja a coordenadas del frame
        c = c.copy()
        c[:, :, 1] += roi_top + y0
        points.append((x, roi_top + y0 + y))
        contours.append(c)
        ref_x = x

    if not points:
        return None

    # Error = mezcla ponderada de la desviación cercana y lejana respecto al centro,
    # dividida entre la mitad del ancho para dejarla en [-1, 1].
    #
    # La x se evalúa sobre una recta ajustada por mínimos cuadrados a TODOS los
    # centroides de franja, no solo el primero y el último: con N franjas el ruido
    # del error baja como 1/raíz(N) y el centrado de la curva mejora, porque la
    # trayectoria entera pesa en vez de solo sus extremos. El punto cercano real
    # (sin ajustar) se sigue usando para buscar salidas de pista.
    center = w / 2.0
    ys = np.array([p[1] for p in points], dtype=float)
    xs = np.array([p[0] for p in points], dtype=float)
    grado = min(2, len(points) - 1)
    if grado >= 1:
        coef = np.polyfit(ys, xs, grado)
        near_x = float(np.polyval(coef, ys[0]))
        far_x = float(np.polyval(coef, ys[-1]))
    else:
        near_x = far_x = xs[0]
    error_px = near_weight * (near_x - center) + (1.0 - near_weight) * (far_x - center)
    error = float(np.clip(error_px / center, -1.0, 1.0))

    # Ancho de la línea cerca del robot = ancho del contorno de la franja inferior
    line_w = cv2.boundingRect(contours[0])[2]
    exits, reach, others, comp = find_exits(full, points[0][0], points[0][1], line_w, min_area)
    crossbar = closed_crossbar(comp, line_w, points) if comp is not None else False
    floor = pixel_to_floor([e[0] for e in exits] + [points[0]], w, h)
    exits = [Exit(px, tuple(f), math.degrees(math.atan2(f[1], f[0])))
             for (px, _), f in zip(exits, floor[:-1])]
    return LineResult(points, contours, binary, roi_top, error, exits, reach, others,
                      crossbar, tuple(floor[-1]))


# ─────────────────────────────────────────────────────────────────────────────
#  SALIDAS DE LA LÍNEA (ramificaciones y callejones)
# ─────────────────────────────────────────────────────────────────────────────
def find_exits(binary, near_x, near_y, line_w, min_area):
    """
    Cuenta por dónde sale del frame la mancha de línea que se está siguiendo.

    Se usa el frame completo (no solo la ROI) porque ahí se ve si un brazo
    realmente continúa: una barra transversal corta cruza el borde de la ROI
    pero termina dentro de la imagen, así que no es una rama.

    1. Se toma solo el contorno que contiene al punto cercano (near_x, near_y),
       para que una mancha suelta no cuente como rama, y se pinta relleno.
    2. Se recorre el borde del frame como una sola tira 1-D:
         lado izquierdo (abajo → arriba) + borde superior (izq. → der.)
         + lado derecho (arriba → abajo).
       Así una línea que cruza una esquina queda como UN tramo continuo.
    3. Cada tramo blanco de al menos MIN_EXIT_PX píxeles es una salida; su
       punto medio es el punto de salida. Dos tramos separados por menos de
       EXIT_MERGE_RATIO × ancho de la línea son la misma cinta (un brillo la
       corta). Lo que toca los lados muy abajo (ENTRY_SIDE_RATIO) es por donde
       entra la línea (o el brazo de una barra que se está pasando), no una rama.

    Devuelve ([(punto_frame, None), ...], reach, others, mask) donde
    reach = altura que alcanza la mancha / alto del frame (1.0 = llega al borde
    superior), others = cuántas manchas más de al menos min_area hay en el
    frame y mask = la mancha seguida, rellena (para closed_crossbar).
    """
    rh, rw = binary.shape[:2]
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [], 0.0, 0, None
    # pointPolygonTest > 0 dentro, = 0 en el borde, < 0 fuera (distancia con signo).
    # Se elige el de mayor valor: el que contiene al punto o, si ninguno, el más cercano.
    pt = (float(near_x), float(near_y))
    comp = max(cnts, key=lambda c: cv2.pointPolygonTest(c, pt, True))
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [comp], -1, 255, cv2.FILLED)

    _, top_y, _, _ = cv2.boundingRect(comp)
    reach = (rh - top_y) / float(rh)
    others = sum(1 for c in cnts if c is not comp and cv2.contourArea(c) >= min_area)

    # Tira del borde con las coordenadas (x, y) de cada píxel. Se usan 2 px de
    # grosor (max) para no perder una línea que roza el borde.
    entry_y = int(rh * (1.0 - config.ENTRY_SIDE_RATIO))
    ys_left = np.arange(entry_y - 1, -1, -1)
    xs_top = np.arange(rw)
    ys_right = np.arange(0, entry_y)
    strip = np.concatenate([mask[ys_left, :2].max(axis=1),
                            mask[:2, :].max(axis=0),
                            mask[ys_right, rw - 2:].max(axis=1)]) > 0
    coords = ([(0, int(y)) for y in ys_left] + [(int(x), 0) for x in xs_top]
              + [(rw - 1, int(y)) for y in ys_right])

    # Tramos de True consecutivos: diferencia de la tira con un 0 a cada lado
    # (+1 = empieza un tramo, -1 = termina)
    d = np.diff(np.concatenate([[0], strip.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = []
    for a, b in zip(starts, ends):
        # Hueco medido en píxeles reales entre el final del tramo anterior y el
        # inicio de este (en una esquina del frame la tira "dobla": así una curva
        # que sale justo por la esquina no se parte en dos salidas)
        if runs and math.dist(coords[a], coords[runs[-1][1] - 1]) < config.EXIT_MERGE_RATIO * line_w:
            runs[-1][1] = b            # Hueco más angosto que la cinta: mismo tramo
        else:
            runs.append([a, b])
    exits = []
    for a, b in runs:
        if b - a < config.MIN_EXIT_PX:
            continue
        x, y = coords[(a + b - 1) // 2]
        exits.append(((x, y), None))

    # Si la mancha no toca el borde inferior, la línea entra por un costado
    # (cruza la imagen en diagonal): la salida más cercana al punto cercano
    # es en realidad la entrada y no cuenta como rama.
    if exits and not mask[rh - 2:, :].any():
        dist = [math.hypot(x - near_x, y - near_y) for (x, y), _ in exits]
        exits.pop(int(np.argmin(dist)))
    return exits, reach, others, mask


def closed_crossbar(mask, line_w, points):
    """
    ¿Se ve una barra transversal que NO es camino?

    La barra es perpendicular a la línea, pero si el robot va en curva se ve
    en diagonal. Por eso primero se ROTA la máscara (cv2.warpAffine) para
    dejar el tronco vertical: el ángulo sale de la dirección entre el primer y
    el último centroide seguido, atan2(dx, −dy). Así la barra queda horizontal.
    Se rota también una máscara "válida" (todo 1) para saber qué es imagen y
    qué quedó fuera del frame al girar.

    Luego, fila por fila, se miden los tramos blancos (diferencia horizontal:
    +1 = empieza, −1 = termina). Una fila es barra si:
      1. su tramo mide al menos CROSSBAR_WIDTH_RATIO × ancho de la línea;
      2. es delgada: 1.5 anchos más abajo está el tronco (un solo tramo angosto)
         y 1.5 anchos más arriba la línea sigue como UN solo tramo angosto
         (en una Y arriba habría dos tramos: dos caminos);
      3. sobresale del tronco hacia LOS DOS lados al menos un ancho de línea
         (una T hacia un lado no sobresale del otro);
      4. al menos uno de sus extremos termina DENTRO de la imagen (el píxel
         siguiente es piso válido): desde la cámara se ve que ese brazo no
         lleva a ningún lado.
    Cerca de la cámara los brazos se salen por los costados y ya no se
    distinguirían de ramas, por eso Brain recuerda la barra (por odometría).
    """
    rh, rw = mask.shape[:2]
    (x0, y0), (x1, y1) = points[0], points[-1]
    angle = math.degrees(math.atan2(x1 - x0, y0 - y1)) if len(points) > 1 else 0.0
    # getRotationMatrix2D gira antihorario con ángulo > 0; un tronco inclinado
    # hacia la derecha (angle > 0) se endereza girándolo antihorario
    rot = cv2.getRotationMatrix2D((rw / 2.0, rh / 2.0), angle, 1.0)
    m = cv2.warpAffine(mask, rot, (rw, rh), flags=cv2.INTER_NEAREST) > 0
    valid = cv2.warpAffine(np.full((rh, rw), 255, np.uint8), rot, (rw, rh),
                           flags=cv2.INTER_NEAREST) > 0
    if not m.any():
        return False
    top_y = int(np.flatnonzero(m.any(axis=1))[0])

    d = np.diff(np.pad(m, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    rows, xs = np.nonzero(d == 1)          # Orden por filas: inicios y finales
    _, xe = np.nonzero(d == -1)            # quedan emparejados índice a índice
    by_row = {}
    for r, a, b in zip(rows.tolist(), xs.tolist(), xe.tolist()):
        by_row.setdefault(r, []).append((a, b))

    # Ancho real de la línea ya enderezada: la mediana del ancho de los tramos
    # (casi todas las filas cortan solo el tronco). El line_w recibido se mide
    # en horizontal y en una diagonal sale inflado.
    widths = xe - xs
    lw = max(float(np.median(widths)) if len(widths) else line_w, 3.0)
    step = int(1.5 * lw)
    gap = 3                                # Píxeles más allá del extremo que se miran
    narrow = lambda runs, a, b: [(u, v) for u, v in runs
                                 if v > a and u < b and v - u < 2 * lw]
    inside = lambda r, x: 0 <= x < rw and valid[r, x]
    for r in range(top_y + step, rh - step):
        for a, b in by_row.get(r, []):
            if b - a < config.CROSSBAR_WIDTH_RATIO * lw:
                continue
            # Extremo "a la vista": justo después del tramo todavía hay imagen (piso)
            if not (inside(r, a - gap) or inside(r, b + gap)):
                continue                      # sale de la imagen por los dos lados
            below = narrow(by_row.get(r + step, []), a, b)
            above = narrow(by_row.get(r - step, []), a, b)
            if len(below) != 1 or len(above) != 1:
                continue
            tl, tr = below[0]                 # tronco bajo la barra
            if tl - a >= lw and b - tr >= lw:
                return True
    return False


# Caché de la homografía imagen → piso (depende solo del tamaño del frame)
_FLOOR_H = {}


def pixel_to_floor(points, w, h):
    """
    Pasa puntos de la imagen (x, y) a cm sobre el piso (adelante, derecha).

    La cámara ve un trapecio de piso (config.SIM_CAM_*): angosto cerca y
    ancho lejos. Las 4 esquinas del frame corresponden a las 4 esquinas de
    ese trapecio, y con esas 4 parejas cv2.getPerspectiveTransform resuelve
    la homografía H (3x3). Cada punto se transforma como
        [X, Y, W]ᵀ = H · [x, y, 1]ᵀ    →    (X/W, Y/W)
    que es la división que hace cv2.perspectiveTransform.
    """
    if (w, h) not in _FLOOR_H:
        near = config.SIM_CAM_NEAR_CM
        far = near + config.SIM_CAM_DEPTH_CM
        nw, fw = config.SIM_CAM_NEAR_WIDTH_CM / 2, config.SIM_CAM_FAR_WIDTH_CM / 2
        img = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        floor = np.float32([[far, -fw], [far, fw], [near, nw], [near, -nw]])
        _FLOOR_H[(w, h)] = cv2.getPerspectiveTransform(img, floor)
    pts = np.float32(points).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, _FLOOR_H[(w, h)]).reshape(-1, 2)
