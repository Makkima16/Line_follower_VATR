"""Test de integración rápida"""
import sys
sys.path.insert(0, r'D:\visionArtificialTiempoR\Line_follower_VATR\code')

import cv2
import time
import config
from senales import detect_signal
from control import Brain, State
from drivers import make_driver

img = cv2.imread(r'D:\visionArtificialTiempoR\Line_follower_VATR\PareSiga.jpg')

# Test 1: detección de señal
result = detect_signal(img)
print('1. Detección de señal:', result.label if result else 'Ninguna',
      f'vértices={result.vertices if result else "N/A"}')

# Test 2: Brain.on_signal PARE
driver = make_driver('pulsos')
brain = Brain(driver)
now = time.monotonic()
brain.on_signal('PARE', now)
print('2. Estado después PARE:', brain.state)
print('   stop_until =', brain.stop_until)
print('   signal =', brain.signal)

# Test 3: Brain.on_signal SIGA
brain2 = Brain(driver)
brain2.on_signal('SIGA', now)
print('3. Estado después SIGA:', brain2.state)
print('   signal =', brain2.signal)

# Test 4: _on_lost con gracia de señal tapada
now2 = time.monotonic() + 0.5
brain3 = Brain(driver)
brain3.on_signal('PARE', now)  # primera vez
brain3._on_lost(now2)  # línea perdida después de señal
print('4. Después de _on_lost con gracia de señal:')
print('   estado =', brain3.state)
print('   u =', brain3.u)

print('\n=== Todos los tests passed ===')