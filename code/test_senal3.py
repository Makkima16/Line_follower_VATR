"""Test final de detección de señales: ambos octágonos"""
import sys
sys.path.insert(0, r'D:\visionArtificialTiempoR\Line_follower_VATR\code')

import cv2
import numpy as np
from senales import detect_signal, SignalResult

img = cv2.imread(r'D:\visionArtificialTiempoR\Line_follower_VATR\PareSiga.jpg')
result = detect_signal(img)
if result is None:
    print('Ninguna señal detectada')
else:
    print(f'Señal: {result.label}')
    print(f'  Vértices: {result.vertices}')
    print(f'  Área: {result.area:.0f} px')
    print(f'  BBox: {result.bbox}')
    print(f'  Centroide: {result.center}')

    # Dibujar resultado
    out = img.copy()
    cv2.drawContours(out, [result.contour], -1, (0, 255, 0), 2)
    cx, cy = int(result.center[0]), int(result.center[1])
    cv2.circle(out, (cx, cy), 7, (0, 0, 255), -1)
    cv2.putText(out, result.label, (cx - 30, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Señal detectada", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()