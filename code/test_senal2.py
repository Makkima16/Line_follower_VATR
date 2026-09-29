"""Test directo de la lógica de detección de señales"""
import sys
sys.path.insert(0, r'D:\visionArtificialTiempoR\Line_follower_VATR\code')

import cv2
import numpy as np
from config import (RED_HUE1_LOW, RED_HUE1_HIGH, RED_HUE2_LOW, RED_HUE2_HIGH,
                    GREEN_HUE_LOW, GREEN_HUE_HIGH, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL,
                    MIN_SIGNAL_AREA)

img = cv2.imread(r'D:\visionArtificialTiempoR\Line_follower_VATR\PareSiga.jpg')
h, w = img.shape[:2]
frame = img  # already BGR

print(f"Frame shape: {h}x{w}")
print(f"MIN_SIGNAL_AREA={MIN_SIGNAL_AREA}, threshold={MIN_SIGNAL_AREA*w*h:.0f}")

# Probar SIGA (verde) paso a paso
label = "SIGA"
print(f"\n--- Probando label={label} ---")
mask = cv2.inRange(frame,
    np.array([GREEN_HUE_LOW, SIGNAL_MIN_SAT, SIGNAL_MIN_VAL], dtype=np.uint8),
    np.array([GREEN_HUE_HIGH, 255, 255], dtype=np.uint8))
print(f"Máscara raw px: {mask.sum()/255:.0f} ({100*mask.sum()/(w*h):.2f}%)")

k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
print(f"Máscara después de morph px: {mask.sum()/255:.0f}")

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contours after morph: {len(cnts)}")
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    ok = area >= MIN_SIGNAL_AREA * w * h
    print(f"  Contour {i}: area={area:.0f} threshold={MIN_SIGNAL_AREA*w*h:.0f} -> {'PASS' if ok else 'FAIL'}")
    if ok:
        perimeter = cv2.arcLength(c, True)
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(c, epsilon, True)
        print(f"    vertices={len(approx)}, perimeter={perimeter:.0f}, epsilon={epsilon:.2f}")
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        print(f"    solidity={solidity:.3f}")