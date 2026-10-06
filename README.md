# Robot seguidor de línea con visión artificial

Proyecto del reto *Visión Artificial en Tiempo Real*. Un robot móvil (mBot de
Makeblock, placa ATmega) sigue una línea negra usando **solo visión clásica y
matemáticas**: sin redes neuronales, modelos pre-entrenados ni cajas negras.

La cámara es un **celular montado en el robot** que manda el video al PC por
WiFi. El PC procesa cada frame con OpenCV, calcula cuánto girar con un control
PD y le envía al robot comandos cortos por **Bluetooth**.

```
 Celular (cámara)  ──WiFi──►  PC: visión → control PD → driver  ──Bluetooth──►  Robot
      ▲                                                                            │
      └──────────────────── el robot gira y la cámara ve otra cosa ◄───────────────┘
```

---

## Estructura

```
Seguidor/
├── README.md
├── requirements.txt
├── pistas/                 Fotos cenitales de pistas para el simulador
│   ├── zigzag.jpeg
│   ├── signals.jpeg        Bifurcación en Y con hexágono rojo (PARE) en el tronco
│   └── ramificacion.png    Pista sintética: PARE + bifurcación con una rama sin salida
└── code/
    ├── main.py             Punto de entrada: cámara real + robot (o robot simulado)
    ├── simulador.py        Simulador de lazo cerrado sobre una foto de la pista
    ├── config.py           TODAS las constantes (cámara, visión, PD, robot, simulador)
    ├── vision.py           Detección de la línea: umbral, morfología, contornos, centroides
    ├── control.py          Control PD + máquina de estados (siguiendo / buscando / PARE / media vuelta...)
    ├── mapa.py             Odometría + mapa de cruces + exploración en profundidad (DFS)
    ├── senales.py          Señales: hexágono/octágono rojo (PARE), octágono verde (SIGA)
    ├── drivers.py          Traduce el mando u ∈ [-1, 1] al protocolo del robot
    ├── robot.py            Conexión (Bluetooth RFCOMM o serie) + hilo de envío
    ├── sim.py              Ventana "Movimiento" (trayectoria según los comandos enviados)
    ├── phone_camera.py     Servidor HTTP/HTTPS que recibe el video del celular
    └── templates/
        └── camera.html     Página que abre el celular para transmitir la cámara
```

---

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`openssl` debe estar instalado: se usa para el certificado HTTPS que exige el
navegador del celular para dar acceso a la cámara.

---

## Uso

Todos los comandos se ejecutan dentro de `code/`.

### 1. Con la cámara del celular

```bash
cd code
../.venv/bin/python main.py                          # celular + robot simulado
../.venv/bin/python main.py --robot 00:1B:10:21:2C:1B  # mBot real por Bluetooth
../.venv/bin/python main.py --robot /dev/rfcomm0 --driver velocidades
```

1. El PC y el celular deben estar en la **misma red WiFi**.
2. En el celular abre la URL que imprime la terminal (`http://<IP-del-PC>:8080`).
   La página te lleva a la versión HTTPS. Acepta la advertencia del certificado
   (es auto-firmado) y da permiso a la cámara.
3. Monta el celular en el robot **mirando hacia adelante y hacia el piso**, de modo que la
   línea entre por la parte de abajo de la imagen.

> **Si el celular no carga la página:** la IP del PC pudo haber cambiado (DHCP).
> Usa siempre la que imprime la terminal y vuelve a aceptar el certificado.
> Algunas redes (universidad, hoteles) aíslan a los clientes: en ese caso usa el
> hotspot del celular.

Otras fuentes de video: `--camara usb` (webcam del PC) o `--video pista.mp4`.

### 2. Simulador (sin robot)

```bash
cd code
../.venv/bin/python simulador.py ../pistas/zigzag.jpeg
../.venv/bin/python simulador.py ../pistas/zigzag.jpeg --ancho-cm 200 --driver velocidades
../.venv/bin/python simulador.py ../pistas/ramificacion.png    # PARE + ramificación + callejón
```

Toma una **foto cenital** de la pista como piso y pone encima un robot virtual.
En cada frame recorta lo que vería su cámara, lo pasa por el mismo `vision.py`
y `control.py` del robot real y mueve el robot con los comandos que salen. Así
se puede calibrar Kp, Kd, umbral, etc. sin tener el robot.

- `--ancho-cm`: cuántos centímetros reales representa el ancho de la foto (escala).
- Clic en el mapa: posición inicial; segundo clic: hacia dónde mira.

> Mover el celular con la mano sobre la pista **no** prueba el control, porque
> los giros que ordena el programa no mueven la cámara (lazo abierto). Para eso
> está el simulador.

### Teclas

| Tecla | Acción |
|---|---|
| `q` | Salir |
| `espacio` | Pausar / reanudar (el robot se detiene) |
| `r` | Reiniciar trayectoria (o simulación) |
| `+` / `-` | Acelerar / frenar la simulación (solo `simulador.py`) |

### Ventanas

- **Camara / Camara virtual:** contornos de la línea (verde), centroides por franja
  (rojo), estado, error `e`, mando `u` (barra inferior) y último comando enviado.
- **Linea (binaria):** resultado del umbral. La línea debe verse blanca y limpia.
- **Calibracion:** trackbars para ajustar en vivo umbral, ROI, peso cercano, Kp, Kd,
  zona muerta y giros máx. / velocidad base.
- **Movimiento** (`main.py`) / **Mapa** (`simulador.py`): trayectoria del robot.
- **Mapa (odometria):** lo que el robot *recuerda*: su rastro estimado, cada cruce
  (`N0`, `N1`…; amarillo = en la pila) y sus ramas: blanca = pendiente, verde =
  explorando, roja = sin salida, azul = origen.

---

## Cómo funciona

### 1. Visión (`vision.py`)

1. **Orientación y escala:** se rota/espeja según el montaje y se reduce a 320 px de
   ancho para procesar en tiempo real.
2. **Segmentación:** se umbraliza el canal **V = max(R, G, B)** (el V de HSV). Un
   píxel solo es "negro" si es oscuro en los tres canales, así que los objetos de color
   no se confunden con la línea. El umbral es fijo (trackbar) u Otsu si vale 0.
3. **Morfología:** un *cierre* rellena los brillos dentro de la cinta y una *apertura*
   borra puntos sueltos. Un segundo umbral más permisivo (`LINE_WEAK_MARGIN`) añade la
   cinta lejana, fina y más clara, solo donde está unida a la línea ya detectada.
4. **ROI por franjas:** solo se analiza la parte **superior** del frame (el piso delante
   del robot; abajo se ve el propio chasis y se recorta), cortada en 5 franjas horizontales. En cada una se buscan contornos
   (`cv2.findContours`) y se calcula su centroide con momentos:
   `x = m10 / m00`, `y = m01 / m00`.
   Subiendo de franja en franja se elige el trozo más cercano al anterior, lo que
   permite seguir la línea en curvas e ignorar manchas.
5. **Error geométrico:** mezcla el punto cercano (dónde está la línea ahora) y el lejano
   (hacia dónde va la curva), normalizado a `[-1, 1]` respecto al centro de la imagen:
   `e = [w·(x_cerca − cx) + (1−w)·(x_lejos − cx)] / cx`.
   `e > 0` significa que la línea está a la derecha.

### 2. Control (`control.py`)

- **PD:** `u = Kp·e + Kd·de/dt`, saturado a `[-1, 1]`. La derivada (filtrada con una
  media exponencial) frena el giro cuando el error ya se está corrigiendo y evita
  pasarse de la línea.
- **Estados:**
  - `SIGUIENDO`: línea visible, se aplica el PD.
  - `BUSCANDO`: la línea se perdió más de `LOST_GRACE_S`. Gira hacia el último lado
    donde se vio (failsafe, objetivo: 0 intervenciones).
  - `DETENIDO`: sin línea durante `LOST_STOP_S`. Se para por seguridad.
  - `PARE`: hexágono u octágono rojo confirmado `SIGNAL_CONFIRM_FRAMES` frames. Se
    detiene `STOP_DURATION_S` y reanuda solo (o antes, si ve un octágono verde SIGA).
    La señal no vuelve a actuar hasta que deja de verse `SIGNAL_COOLDOWN_S`.
  - `MEDIA VUELTA`: callejón sin salida; gira en sitio hasta volver a la línea.
  - `FIN`: ya no queda ninguna rama por explorar.
- Todos los tiempos usan `time.monotonic()`, así que nada bloquea el video.

### 3. Ramificaciones y mapa (`vision.find_exits`, `mapa.py`)

- **Salidas:** se toma la mancha de línea que se sigue y se recorre el borde de la ROI
  (izquierda ↑, arriba →, derecha ↓) como una tira 1-D. Cada tramo blanco es una
  **salida**: 1 = tramo normal, 2+ = **ramificación**, 0 con la línea terminando en la
  mitad cercana y sin otro trozo a la vista = **callejón sin salida**.
- **Píxeles → cm:** con `cv2.getPerspectiveTransform` (las 4 esquinas del frame ↔ el
  trapecio de piso `SIM_CAM_*`) cada salida se pasa a cm y se obtiene el **rumbo** de la
  rama: `rumbo = θ − atan2(derecha, adelante)`.
- **Odometría:** sin encoders, la pose se integra con los mismos comandos enviados
  (`x += v·cosθ·dt`, `y += v·sinθ·dt`, `θ += ω·dt`).
- **Mapa y DFS:** cada cruce es un nodo con sus ramas (pendiente / explorando /
  sin salida / origen). En un cruce nuevo se toma una rama según `BRANCH_POLICY`; en un
  callejón se marca la rama, media vuelta y se regresa al último cruce de la pila, donde
  se toma la siguiente rama pendiente. Si el cruce se agota, se sale por su origen y se
  sigue retrocediendo. Al reconocer un cruce se corrige la deriva de posición y rumbo.

### 4. Drivers (`drivers.py`): de `u` a comandos

- **`pulsos`** (firmware del curso, letras `w/a/d/x`, pulsos de duración fija): como no
  hay control de velocidad, el giro proporcional se consigue en el tiempo. Por cada
  avance `w` se intercalan en promedio `|u|·MAX_TURN_PULSES` giros, acumulando la parte
  fraccionaria (*sigma-delta*). Ejemplo: `u=0.2 → wdwdwdwdwwd…`, `u=0.5 → ddwddw…`.
  Es un PWM a nivel de comandos, no un zig-zag binario.
- **`velocidades`** (firmware con velocidad por rueda, p. ej. `V:120,80\n`):
  `izq = base·(1+u)`, `der = base·(1−u)`, saturadas a `[0, MAX_SPEED]` y enviadas
  cada `SPEED_PERIOD_S`.

### 5. Concurrencia (`robot.py`, `main.py`)

- **Hilo principal:** cámara → visión → `Brain.on_frame()` → ventanas.
- **Hilo `CommandSender`:** `Brain.next_command()` → driver → Bluetooth, al ritmo que
  marca el driver. La cámara nunca espera al Bluetooth ni al revés.
- Al salir se envía siempre el comando de parada.

### 6. Simulador (`simulador.py`)

La cámara del celular mira adelante y abajo, así que ve un **trapecio** de piso
(angosto cerca, ancho lejos). Con la pose del robot `(x, y, θ)` se calculan las 4
esquinas de ese trapecio sobre la foto. `cv2.getPerspectiveTransform` obtiene la
homografía que las lleva a las esquinas del frame y `cv2.warpPerspective` genera la
imagen virtual. El robot se mueve con la cinemática de un robot diferencial
(`x += v·cosθ·dt`, `y += v·sinθ·dt`, `θ += ω·dt`), con latencia de cámara y un reloj
simulado para que cada corrida sea reproducible.

---

## Calibración

Todo está en `code/config.py`. Los parámetros marcados con `[TB]` también se ajustan
en vivo con los trackbars.

| Parámetro | Efecto |
|---|---|
| `BINARY_THRESHOLD` | Umbral de "negro". Súbelo si la línea sale cortada; bájalo si aparecen sombras. `0` = Otsu. |
| `ROI_HEIGHT_RATIO` | Fracción **superior** del frame analizada (línea cian). Bájala si entra el robot o su sombra. |
| `CHASSIS_MASK_RATIO` | Fracción inferior del frame que es el propio robot (línea roja); nunca se analiza. |
| `LINE_WEAK_MARGIN` | Margen del umbral débil (doble umbral): rescata la cinta lejana unida a la línea firme. |
| `NEAR_WEIGHT` | 1 = solo mira cerca (reacciona tarde). Menor = anticipa las curvas. |
| `KP` | Fuerza del giro. Si oscila en rectas, bájalo; si se sale en curvas, súbelo. |
| `KD` | Amortiguación. Súbelo si se pasa de la línea después de cada curva. |
| `DEADBAND` | Errores más pequeños que esto se ignoran (va recto). |
| `RECOVERY_U`, `LOST_*` | Comportamiento cuando pierde la línea. |
| `PARE_VERTICES`, `SIGNAL_MIN_MEAN_SAT`, `MIN_SIGNAL_AREA` | Qué cuenta como señal PARE (vértices, color, tamaño). |
| `BRANCH_POLICY` | En qué orden se prueban las ramas: `izquierda`, `derecha` o `recto`. |
| `DEAD_END_*`, `JUNCTION_CONFIRM_FRAMES` | Qué tan seguro debe estar para declarar callejón / cruce. |
| `SIM_CAM_*` | Trapecio de piso que ve la cámara. **Medirlo en el robot real**: lo usa el mapa. |
| `SIM_FORWARD_CM_PER_S`, `SIM_TURN_DEG_PER_S` | Base de la odometría. Calibrar: mandar `w`/`a` N veces y medir cm y grados reales. |

### Al recibir el robot (sección **ROBOT** de `config.py`)

1. **Conexión:** `ROBOT_ADDRESS` = MAC del mBot (`"AA:BB:CC:DD:EE:FF"`) o puerto serie
   (`"/dev/rfcomm0"`, `"COM5"`). Con `None` el robot es simulado.
2. **Protocolo:** `DRIVER = "pulsos"` o `"velocidades"`. Ajusta `PULSE_COMMANDS`,
   `PULSE_*_S` o `SPEED_FORMAT` según el firmware. Para otro protocolo, crea una clase
   en `drivers.py` con `next()` y `stop_message()` y regístrala en `DRIVERS`.
3. **Montaje de la cámara:** `ROTATE` y `FLIP_HORIZONTAL`. Comprueba que con la línea a
   la derecha de la imagen, `e` salga positivo.
4. **Sentido de giro:** si gira al revés, intercambia `left`/`right` en `PULSE_COMMANDS`.
5. **Simulador:** mide qué trozo de piso ve la cámara montada y actualiza `SIM_CAM_*`.

---

## Restricciones del reto

Solo se usan Python, OpenCV, NumPy, `threading` y `pyserial`. Toda la lógica
(umbral, momentos, centroides, PD, estados) es explícita y está implementada en este
repositorio. No se usa Deep Learning, cascadas Haar, APIs externas ni librerías que
resuelvan el seguimiento como caja negra.
