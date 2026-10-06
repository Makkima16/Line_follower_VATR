"""Script de prueba rápida de senales.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from config import (RED_HUE1_LOW, RED_HUE1_HIGH, RED_HUE2_LOW, RED_HUE2_HIGH,
                    GREEN_HUE_LOW, GREEN_HUE_HIGH, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL,
                    MIN_SIGNAL_AREA)

from senales import detect_signal, SignalResult

img = cv2.imread(str(Path(__file__).parent.parent / 'PareSiga.jpg'))
print(f"Imagen cargada: {img.shape}")

result = detect_signal(img)
if result is None:
    print(">>> Ninguna señal detectada <<<")
    # Debug: probar verde por separado
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = np.array([GREEN_HUE_LOW, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL], dtype=np.uint8)
    high = np.array([GREEN_HUE_HIGH, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    print(f"Green mask px: {mask.sum()/255:.0f} ({100*mask.sum()/(w*h):.2f}%)")
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contours after morph: {len(cnts)}")
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        print(f"Max area: {area:.0f} vs min: {MIN_SIGNAL_AREA*w*h:.0f}")
        perimeter = cv2.arcLength(c, True)
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(c, epsilon, True)
        print(f"Green vertices: {len(approx)}")
else:
    print(f"Señal: {result.label}, vertices={result.vertices}, area={result.area:.0f}")
    out = img.copy()
    cv2.drawContours(out, [result.contour], -1, (0, 255, 0), 2)
    cx, cy = int(result.center[0]), int(result.center[1])
    cv2.circle(out, (cx, cy), 7, (0, 0, 255), -1)
    cv2.putText(out, result.label, (cx - 30, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()