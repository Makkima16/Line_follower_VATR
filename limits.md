# Contexto del Proyecto
Actúa como un experto en Visión Artificial y revisor de código estricto. Tu tarea es analizar el repositorio "makkima16/line_follower_vatr" (específicamente archivos como `vision.py`, `control.py` y `main.py`) para un proyecto universitario de un robot autónomo seguidor de línea[cite: 7]. 

**Resumen de la misión:** 
El código debe funcionar como el cerebro de un robot móvil con tracción diferencial que se comunica vía Bluetooth[cite: 3, 4]. Debe seguir una línea en el suelo corrigiendo su trayectoria proporcionalmente (sin zig-zag) y utilizar una máquina de estados para detenerse ante un octágono rojo (señal de PARE) y reanudar la marcha ante un octágono verde (señal de SIGA)[cite: 4, 7]. Si pierde la línea, debe tener una rutina de auto-recuperación (0 intervenciones humanas)[cite: 4, 6]. Todo esto debe ejecutarse de forma concurrente para no afectar los FPS del video[cite: 4].

Para este reto, **solo** se pueden utilizar técnicas de visión artificial clásica enseñadas en el curso. 

---

# 🛑 RESTRICCIONES ESTRICTAS (LO QUE NO SE PUEDE USAR)
Revisa línea por línea y lanza una **ALERTA CRÍTICA** si el código importa, implementa o hace uso de cualquiera de las siguientes tecnologías:
- Redes neuronales artificiales o Deep Learning[cite: 5, 7].
- Modelos previamente entrenados[cite: 5, 7].
- Detectores basados en YOLO, SSD, Faster R-CNN u otros métodos de IA modernos[cite: 7].
- Cascadas Haar (Haar Cascades)[cite: 5, 7].
- Servicios externos de inteligencia artificial (APIs de visión)[cite: 7].
- Algoritmos, funciones o bibliotecas que realicen automáticamente la detección de la línea o las señales sin que el equipo haya implementado la lógica matemática/geométrica subyacente[cite: 4, 7].

---

# ✅ HERRAMIENTAS Y TÉCNICAS PERMITIDAS (LO QUE SÍ SE DEBE USAR)
Verifica que la solución dependa única y exclusivamente de las siguientes técnicas y librerías base (`cv2`, `numpy`, `pyserial`, `threading`)[cite: 2, 3, 6]:
1. **Manejo de imágenes:** Operaciones lógicas (AND, OR) y aritméticas (resta absoluta de imágenes `cv2.absdiff`)[cite: 7, 10, 11].
2. **Espacios de color:** Conversión y segmentación usando RGB, HSV o CIELab (`cv2.cvtColor`)[cite: 7, 9, 10].
3. **Transformaciones:** Recorte de regiones de interés (ROI), redimensionamiento y rotación de imágenes[cite: 7].
4. **Filtros y Suavizado:** Reducción de ruido utilizando filtros como el Gaussiano (`cv2.GaussianBlur`)[cite: 7, 10].
5. **Segmentación:** Umbralización (Thresholding) y algoritmos básicos como K-Means[cite: 7, 8].
6. **Morfología:** Operaciones morfológicas como erosión, dilatación, apertura y cierre[cite: 7, 10].
7. **Detección de características:** Detección de bordes mediante el algoritmo de Canny (`cv2.Canny`) y detección de contornos (`cv2.findContours`)[cite: 7, 10].
8. **Análisis Geométrico:** Identificación de formas geométricas simples y evaluación de propiedades de los contornos (área, perímetro, centroide/momentos, aproximación poligonal con `cv2.approxPolyDP`, relación de aspecto y rectángulos delimitadores `cv2.boundingRect`)[cite: 2, 7, 10].
9. **Control Motor:** Lazo de control clásico (Proporcional o PD/PID) calculando el error a partir del centroide de la línea, con saturación de velocidades[cite: 4, 6].

---

# Instrucciones de Revisión
Por favor, analiza el código del repositorio proporcionado e indícame:
1. **Cumplimiento de Restricciones:** ¿Existe alguna línea de código, librería importada o método que viole las restricciones estrictas mencionadas? Sé muy específico.
2. **Uso Adecuado de Herramientas:** ¿Se están utilizando correctamente las técnicas permitidas (por ejemplo, ¿`approxPolyDP` está configurado para buscar 8 lados en las señales?)?
3. **Lógica de Control (Integrante 3):** ¿El código implementa adecuadamente la concurrencia (hilos) para Bluetooth/Video, el cálculo de error proporcional y la máquina de estados (PARE/SIGA/Línea)?