## Visión Artificial Reto 1

FECHA ELABORACIÓN DEL

DOCUMENTO: 22/Sep/2026

Profesor: Felipe Buitrago Carmona

## Reto de visión artificial: el cerebro de un robot autónomo seguidor de línea

Los estudiantes asumirán el papel de un equipo de ingenieros encargado de programar el “cerebro” de un robot móvil. El robot contará con una cámara conectada y deberá desplazarse de manera autónoma sobre una pista que contiene una línea guía.

La misión consiste en desarrollar un algoritmo de visión artificial capaz de analizar, en tiempo real, las imágenes capturadas por la cámara y tomar decisiones para controlar el movimiento del robot. El sistema deberá permitir que el robot siga la línea trazada sobre la pista sin descarrilarse y responda correctamente ante dos señales de tránsito que podrán aparecer durante el recorrido:

- Octágono rojo: señal de PARE. El robot deberá detenerse durante el tiempo establecido por el docente.

- Octágono verde: señal de SIGA. El robot deberá continuar su recorrido.

El reto no se limita a lograr que el robot avance. Los estudiantes deberán justificar cómo su algoritmo identifica la línea, determina la dirección del movimiento, reconoce las señales y controla el comportamiento del robot frente a cada situación.


## Visión Artificial Reto 1

FECHA ELABORACIÓN DEL

DOCUMENTO: 22/Sep/2026

Profesor: Felipe Buitrago Carmona

## Objetivo general

Diseñar e implementar un algoritmo de visión artificial que permita a un robot móvil seguir una línea sobre una pista, corregir su trayectoria y reconocer señales representadas mediante octágonos rojos y verdes, sin utilizar técnicas de aprendizaje profundo.

## Objetivos específicos

- 1. Capturar y analizar las imágenes provenientes de la cámara del robot.

- 2. Identificar la línea guía mediante técnicas de procesamiento y segmentación de imágenes.

- 3. Calcular la posición de la línea respecto al centro del robot.

- 4. Generar acciones de control para corregir la trayectoria y evitar el descarrilamiento.

- 5. Detectar e identificar los octágonos rojo y verde ubicados en diferentes puntos de la pista.

- 6. Detener el robot ante la señal de PARE y reanudar su recorrido ante la señal de SIGA.

- 7. Comparar la estrategia desarrollada con las soluciones de los demás equipos y explicar sus ventajas, limitaciones y posibles mejoras.

## Restricciones técnicas

Para resolver el reto, los equipos deberán utilizar únicamente técnicas de procesamiento de imágenes y visión artificial trabajadas durante el curso.

## Se permite el uso de:

- Operaciones lógicas y aritméticas sobre imágenes.

- Conversión entre espacios de color como RGB, HSV y CIELab.

- Recorte de regiones de interés.

- Redimensionamiento y rotación de imágenes.

- Umbralización.

- Segmentación por color.

- K-Means en su modalidad básica.

- Operaciones morfológicas como erosión, dilatación, apertura y cierre.


## Visión Artificial Reto 1

FECHA ELABORACIÓN DEL

DOCUMENTO: 22/Sep/2026

Profesor: Felipe Buitrago Carmona

- Suavizado y reducción de ruido.

- Detección de bordes mediante Canny.

- Detección y análisis de contornos.

- Identificación de formas geométricas simples.

- Propiedades básicas de los contornos, como área, perímetro, centroide, aproximación poligonal y relación de aspecto.

## No se permite utilizar:

- Redes neuronales artificiales.

- Deep learning.

- Modelos previamente entrenados.

- Detectores basados en YOLO, SSD, Faster R-CNN u otros métodos equivalentes.

- Cascadas Haar.

- Servicios externos de inteligencia artificial.

- Algoritmos o bibliotecas que realicen automáticamente la detección de la línea o las señales sin que el equipo implemente la lógica correspondiente.

El código deberá ser comprensible, estar organizado y permitir explicar claramente el funcionamiento de cada etapa del algoritmo.

## Entregables

Cada equipo deberá presentar:

- 1. Un póster o material visual que presente la metodología, las etapas del procesamiento y los resultados obtenidos.

- 2. Una demostración práctica del robot durante la competencia.

## Competencia de robótica

La competencia consistirá en realizar un recorrido completo sobre la pista. Cada equipo podrá realizar los intentos definidos por el docente, de acuerdo con el tiempo disponible y las condiciones establecidas.

## El resultado tendrá en cuenta:

- Tiempo total empleado para completar la pista.

- Cumplimiento de la señal de PARE.


## Visión Artificial Reto 1

FECHA ELABORACIÓN DEL

DOCUMENTO: 22/Sep/2026

Profesor: Felipe Buitrago Carmona

- Cumplimiento de la señal de SIGA.

- Número de descarrilamientos.

- Necesidad de intervención manual.

- Capacidad del robot para recuperar la trayectoria.

## Ayudas

En la siguiente carpeta podrá encontrar videos de ensayo con los cuales podrá experimentar sus algoritmos

https://drive.google.com/drive/folders/1m6mazLjCKPlwaVpH-arGSYZMF_P2KC77?usp=sha ring [URL 🔗](https://drive.google.com/drive/folders/1m6mazLjCKPlwaVpH-arGSYZMF_P2KC77?usp=sharing)

Algoritmos

Analisis

Nada de IA no vista en clase
