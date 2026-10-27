# Explicación del código

## `vision.py` — Detección de la línea

Este módulo convierte cada fotograma de la cámara en un único número: el **error** de la línea, un valor en `[-1, 1]` que indica qué tan desviada está la línea respecto al centro de la imagen. Solo usa visión clásica: umbralización, morfología, contornos y momentos.

### Flujo general

```
frame crudo → prepare_frame() → detect_line() → LineResult (o None) → Brain (control PD)
                                      └─ usa line_mask() internamente
```

`main.py` lo usa así (`main.py:186-189`):

```python
frame = vision.prepare_frame(frame)
line = vision.detect_line(frame, threshold, roi_ratio, near_weight, prev_x)
prev_x = line.points[0][0] if line is not None else None
brain.on_frame(line)
```

`simulador.py` también usa `line_mask()` para generar la pista virtual y `detect_line()` para procesar la cámara simulada.

---

### `_KERNEL5` (línea 12)

```python
_KERNEL5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
```

Elemento estructurante de las operaciones morfológicas: un cuadrado de 5×5 píxeles. Se crea **una sola vez** al importar el módulo para no reconstruirlo en cada fotograma.

---

### Clase `LineResult` (líneas 15-21)

Es un `@dataclass`: un contenedor de datos sin lógica propia. Es lo que devuelve `detect_line()` cuando encuentra la línea.

| Campo | Tipo | Descripción |
|---|---|---|
| `points` | `list` | Centroides `(x, y)`, uno por franja, ordenados de **abajo (cerca del robot) hacia arriba (lejos)**. Están en coordenadas del frame completo. |
| `contours` | `list` | Contorno elegido en cada franja, desplazado a coordenadas del frame. Se usa para dibujar el overlay. |
| `binary` | `np.ndarray` | Máscara binaria de la ROI (línea = 255, blanco). Se muestra en la ventana de la máscara. |
| `roi_top` | `int` | Fila `y` donde empieza la ROI (siempre 0: la ROI es la parte superior del frame). |
| `error` | `float` | **Salida principal.** Error normalizado en `[-1, 1]`: `> 0` → línea a la derecha, `< 0` → línea a la izquierda. Es la entrada del controlador PD. |

---

### Función `prepare_frame(frame)` (líneas 24-36)

Preprocesamiento geométrico: deja el fotograma orientado correctamente y con un tamaño fijo.

**1. Rotación**

```python
rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
       270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(config.ROTATE)
if rot is not None:
    frame = cv2.rotate(frame, rot)
```

Traduce los grados de `config.ROTATE` a la constante de OpenCV. Con `ROTATE = 0`, `.get()` devuelve `None` y no se rota. Permite compensar si el celular queda montado de lado.

**2. Espejo**

```python
if config.FLIP_HORIZONTAL:
    frame = cv2.flip(frame, 1)
```

Voltea la imagen sobre el eje vertical (`1` = izquierda ↔ derecha). Es importante: si la imagen queda espejada, **el signo del error se invierte** y el robot giraría hacia el lado contrario.

**3. Reescalado**

```python
new_h = int(round(h * config.PROCESS_WIDTH / w))
frame = cv2.resize(frame, (config.PROCESS_WIDTH, new_h), interpolation=cv2.INTER_AREA)
```

Reduce el ancho a `PROCESS_WIDTH` (320 px) conservando la proporción de aspecto: `new_h = h · 320 / w`.

- Es la **principal optimización de rendimiento**: un frame de 1280×720 tiene unas 16 veces más píxeles que uno de 320×180, así que todo lo posterior es unas 16 veces más barato.
- `INTER_AREA` promedia los píxeles al reducir, lo que además suaviza ligeramente la imagen.

---

### Función `line_mask(roi, threshold)` (líneas 39-54)

Segmenta la línea negra y devuelve una imagen binaria donde **línea = 255** y **fondo = 0**.

**Línea clave 47: canal V en lugar de escala de grises**

```python
v = cv2.GaussianBlur(roi.max(axis=2), (5, 5), 0)
```

- `roi.max(axis=2)` toma el máximo entre B, G y R de cada píxel, que es exactamente el **canal V de HSV** (`V = max(R, G, B)`). Así se obtiene sin hacer la conversión completa con `cvtColor`, lo que es más rápido.
- **¿Por qué no usar gris?** El gris es un promedio ponderado (`≈ 0.30R + 0.59G + 0.11B`), así que un objeto azul o rojo intenso da un gris bajo y se confundiría con la línea negra. Con V, un píxel solo es oscuro si **los tres canales** son bajos, de modo que los objetos de color vivo no se confunden con la cinta.
- `GaussianBlur` 5×5 reduce el ruido del sensor antes de umbralizar, para evitar píxeles sueltos en la máscara.

**Líneas 48-51: umbralización inversa**

```python
if threshold <= 0:
    _, binary = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
else:
    _, binary = cv2.threshold(v, threshold, 255, cv2.THRESH_BINARY_INV)
```

- `THRESH_BINARY_INV`: `pixel = 255 si V ≤ threshold, si no 0`. Es **inversa** porque lo que interesa es lo oscuro (la cinta).
- **Umbral manual** (`threshold > 0`): se calibra en vivo con la trackbar.
- **Otsu** (`threshold <= 0`): calcula el umbral automáticamente. Recorre el histograma y elige el valor `t` que **maximiza la varianza entre clases** (fondo vs. línea). Es útil cuando cambia la iluminación. En la trackbar, el valor 0 activa Otsu.

**Líneas 53-54: morfología**

```python
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _KERNEL5)
return cv2.morphologyEx(binary, cv2.MORPH_OPEN, _KERNEL5)
```

- **Cierre** (dilatación → erosión): rellena huecos pequeños **dentro** de la cinta, como los reflejos de luz sobre la cinta brillante que la "partirían" en dos.
- **Apertura** (erosión → dilatación): elimina manchas pequeñas del **fondo** (suciedad, sombras chicas) de menos de unos 5 px.
- **El orden importa:** primero se cierra para no perder partes de la línea y después se abre para limpiar el ruido.

---

### Función `detect_line(frame, threshold, roi_ratio=None, near_weight=None, prev_x=None)` (líneas 57-118)

Es el núcleo del módulo. Busca la línea dividiendo la ROI en **franjas horizontales** y encadenando los centroides de abajo hacia arriba.

**Parámetros**

| Parámetro | Descripción |
|---|---|
| `frame` | Fotograma ya preparado con `prepare_frame()`. |
| `threshold` | Umbral de binarización (`0` = Otsu). |
| `roi_ratio` | Fracción superior del frame que se analiza. Si es `None`, se usa `config.ROI_HEIGHT_RATIO`. |
| `near_weight` | Peso del punto cercano frente al lejano en el error. Si es `None`, se usa `config.NEAR_WEIGHT`. |
| `prev_x` | Posición `x` de la línea en el frame anterior (continuidad temporal). |

**Retorna** un `LineResult` si encuentra la línea, o `None` si no la encuentra.

#### 1. Región de interés (líneas 70-72)

```python
band_bottom = min(int(h * roi_ratio), h - int(h * chassis_ratio))
floor = full[:band_bottom, :]
```

Con `roi_ratio = 0.60` se analiza el **60 % superior** del frame. Con el celular montado sobre el robot, la parte inferior es el propio chasis (oscuro y más ancho que la cinta): si entra en la ROI se confunde con la línea. `chassis_ratio` es un segundo tope: esa fracción inferior nunca se analiza. Las filas se **recortan** (no se pintan), y tanto las franjas como la búsqueda de salidas trabajan sobre `floor`.

#### 2. Franjas y límites (líneas 74-79)

```python
slice_h = max(roi_h // config.N_SLICES, 1)
min_area = config.MIN_BLOB_AREA_RATIO * slice_h * w
max_jump = config.MAX_JUMP_RATIO * w
ref_x = prev_x if prev_x is not None else w / 2.0
```

- `slice_h`: alto de cada franja (la ROI se divide en `N_SLICES = 5`).
- `min_area`: un blob que ocupe menos del 1 % de la franja se considera **ruido**.
- `max_jump`: entre una franja y la siguiente, la línea no puede desplazarse más del 25 % del ancho (80 px con 320 px de ancho).
- `ref_x`: referencia para elegir el blob correcto. Parte de `prev_x` (donde estaba la línea en el frame anterior) o, si no existe, del centro de la imagen. Así, si aparecen dos manchas, se elige la que está donde estaba la línea hace un instante.

#### 3. Recorrido de franjas de abajo hacia arriba (líneas 82-84)

```python
for i in range(config.N_SLICES):
    y1 = roi_h - i * slice_h
    y0 = 0 if i == config.N_SLICES - 1 else y1 - slice_h
```

`i = 0` es la franja más baja, la más cercana al robot. La última franja se extiende hasta `y0 = 0` para absorber los píxeles sobrantes cuando `roi_h` no es divisible entre `N_SLICES`.

#### 4. Contornos y centroides (líneas 85-95)

```python
cnts, _ = cv2.findContours(binary[y0:y1, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    m = cv2.moments(c)
    if m["m00"] < min_area:
        continue
    x = m["m10"] / m["m00"]
    dist = abs(x - ref_x)
    if best is None or dist < best[0]:
        best = (dist, x, m["m01"] / m["m00"], c)
```

- `cv2.findContours`: encuentra los blobs blancos de la franja.
  - `RETR_EXTERNAL`: solo los contornos exteriores (ignora los huecos internos).
  - `CHAIN_APPROX_SIMPLE`: comprime los segmentos rectos a sus extremos, lo que ahorra memoria.
- `cv2.moments`: calcula los momentos del contorno.
  - `m00` = **área** del contorno → filtro contra ruido.
  - **Centroide:** `x = m10 / m00`, `y = m01 / m00`. Es el "centro de masa" del blob: `m10` es la suma de las coordenadas `x` y `m00` es el área, así que el cociente da la `x` promedio.
- Se elige el blob cuyo centroide está **más cerca de `ref_x`**, es decir, del punto de la franja anterior.

#### 5. Reglas de parada (líneas 97-103)

```python
if best is None:
    if points:
        break          # La línea termina o sale del cuadro
    continue           # Franja de abajo vacía: probar la siguiente
dist, x, y, c = best
if points and dist > max_jump:
    break              # Salto brusco: es otra mancha
```

| Situación | Acción | Motivo |
|---|---|---|
| Franja vacía y **aún no hay puntos** | `continue` | La línea puede no tocar la parte de abajo (robot algo desviado); se sigue buscando más arriba. |
| Franja vacía y **ya hay puntos** | `break` | La línea terminó o salió del cuadro (curva cerrada). |
| Salto mayor que `max_jump` | `break` | Es otra mancha u otro tramo de la pista, no la continuación de la línea. |

> Nota: la franja más baja **no se valida con `max_jump`** (la condición exige `points` no vacío). Es intencional: permite volver a engancharse a la línea después de perderla.

#### 6. Guardar el punto (líneas 105-109)

```python
c = c.copy()
c[:, :, 1] += roi_top + y0
points.append((x, roi_top + y0 + y))
contours.append(c)
ref_x = x
```

- Se suma `roi_top + y0` a la coordenada `y` del contorno y del centroide para pasar de **coordenadas de la franja** a **coordenadas del frame completo**. Se hace sobre una copia para no alterar el contorno original.
- **`ref_x = x`** es la línea clave del seguimiento: la franja siguiente busca cerca de esta. Así el algoritmo "trepa" por la línea aunque se curve, porque cada franja solo necesita estar cerca de la anterior, no del centro.

#### 7. Cálculo del error (líneas 111-118)

```python
if not points:
    return None

center = w / 2.0
near_x, far_x = points[0][0], points[-1][0]
error_px = near_weight * (near_x - center) + (1.0 - near_weight) * (far_x - center)
error = float(np.clip(error_px / center, -1.0, 1.0))
```

- **Sin puntos → `None`.** El `Brain` lo interpreta como línea perdida y activa la **rutina de recuperación**: gira hacia el último lado conocido (`LOST_GRACE_S`, `RECOVERY_U` en `config.py`).
- **Mezcla cerca/lejos:**

  ```
  error_px = w_near · (near_x − centro) + (1 − w_near) · (far_x − centro)
  ```

  - `near_x` (franja más baja) indica **dónde está** la línea ahora → corrige la posición.
  - `far_x` (franja más alta) indica **hacia dónde va** → **anticipa la curva**, de modo que el robot empieza a girar antes de llegar a ella.
  - Con `NEAR_WEIGHT = 1.0` solo reacciona a lo inmediato (más oscilación); con `0.0` solo mira lejos (tiende a cortar las curvas). El valor por defecto es `0.6`.
- **Normalización:** dividir entre `centro = w/2` hace que `±1` signifique "línea en el borde de la imagen", sin importar la resolución. Por eso `KP` y `KD` no dependen de la cámara.
- `np.clip` asegura el rango `[-1, 1]` como protección.

---

### ¿Por qué franjas y no un solo centroide?

Si se calculara un único centroide de toda la ROI, en una curva o bifurcación ese punto caería en una zona intermedia que puede no estar sobre la línea. Con las franjas encadenadas:

- se sigue la **forma real** de la línea, incluso en curvas;
- se descartan manchas sueltas (filtro de área y de salto máximo);
- se obtiene un punto cercano y uno lejano para **anticipar** las curvas.

### Técnicas usadas (todas permitidas por la rúbrica)

| Técnica | Función de OpenCV / NumPy |
|---|---|
| Rotación, espejo, redimensionado | `cv2.rotate`, `cv2.flip`, `cv2.resize` |
| ROI | Slicing de NumPy `frame[roi_top:, :]` |
| Espacio de color HSV (canal V) | `roi.max(axis=2)` |
| Suavizado gaussiano | `cv2.GaussianBlur` |
| Umbralización (manual / Otsu) | `cv2.threshold` |
| Morfología (cierre, apertura) | `cv2.morphologyEx` |
| Contornos | `cv2.findContours` |
| Momentos y centroides | `cv2.moments` |
