#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from control import State
from policy import SpeedPolicy, apply_signal

fallos = 0


def check(nombre, cond):
    global fallos
    if cond:
        print(f"[PASA] {nombre}")
    else:
        print(f"[FALLA] {nombre}")
        fallos += 1


S = State.SIGUIENDO
P = State.PARE
B = State.BUSCANDO

# 1. PARE desde SIGUIENDO -> estado PARE, sin temporizador (espera a SIGA)
d = apply_signal("PARE", S, SpeedPolicy.NORMAL, 100.0, config)
check("PARE desde SIGUIENDO cambia a PARE", d.new_state == P)
check("PARE no tiene stop_until temporizado", d.stop_until is None)
check("PARE desarma PARE", "PARE" in d.disarm)
check("PARE marca acted", d.acted)

# 2. PARE desde BUSCANDO también funciona
d = apply_signal("PARE", B, SpeedPolicy.NORMAL, 100.0, config)
check("PARE desde BUSCANDO cambia a PARE", d.new_state == P)

# 3. SIGA desde PARE -> reanuda a SIGUIENDO
d = apply_signal("SIGA", P, SpeedPolicy.NORMAL, 101.0, config)
check("SIGA desde PARE reanuda a SIGUIENDO", d.new_state == S)
check("SIGA desde PARE marca acted", d.acted)

# 4. SIGA desde SIGUIENDO -> no cambia estado, no actúa
d = apply_signal("SIGA", S, SpeedPolicy.NORMAL, 102.0, config)
check("SIGA desde SIGUIENDO no cambia estado", d.new_state == S)
check("SIGA desde SIGUIENDO no actúa", not d.acted)

# 5. ZONA ESCOLAR desde SIGUIENDO -> velocidad reducida
d = apply_signal("ZONA ESCOLAR", S, SpeedPolicy.NORMAL, 103.0, config)
check("ZONA ESCOLAR pasa a reduced", d.speed_policy == SpeedPolicy.REDUCED)
check("ZONA ESCOLAR no cambia estado", d.new_state == S)

# 6. ZONA ESCOLAR desde PARE -> prioridad PARE, NO reduce (no reanuda)
d = apply_signal("ZONA ESCOLAR", P, SpeedPolicy.NORMAL, 104.0, config)
check("PARE prioridad: ZONA en PARE no cambia estado", d.new_state == P)
check("PARE prioridad: ZONA en PARE no reduce velocidad", d.speed_policy == SpeedPolicy.NORMAL)

# 7. VELOCIDAD 60 desde reduced -> vuelve a normal
d = apply_signal("VELOCIDAD 60", S, SpeedPolicy.REDUCED, 105.0, config)
check("VELOCIDAD 60 desde reduced vuelve a normal", d.speed_policy == SpeedPolicy.NORMAL)
check("VELOCIDAD 60 desde reduced actúa", d.acted)

# 8. VELOCIDAD 60 desde normal -> conserva normal, no actúa
d = apply_signal("VELOCIDAD 60", S, SpeedPolicy.NORMAL, 106.0, config)
check("VELOCIDAD 60 desde normal conserva normal", d.speed_policy == SpeedPolicy.NORMAL)
check("VELOCIDAD 60 desde normal no actúa", not d.acted)

# 9. VELOCIDAD 60 desde PARE -> prioridad PARE, no reanuda
d = apply_signal("VELOCIDAD 60", P, SpeedPolicy.REDUCED, 107.0, config)
check("PARE prioridad: VELOCIDAD en PARE no reanuda", d.new_state == P)
check("PARE prioridad: VELOCIDAD en PARE no cambia velocidad", d.speed_policy == SpeedPolicy.REDUCED)

# 10. fondo/None -> sin acción
d = apply_signal(None, S, SpeedPolicy.NORMAL, 108.0, config)
check("None no actúa", not d.acted and d.new_state == S)
d = apply_signal("fondo", S, SpeedPolicy.NORMAL, 108.0, config)
check("fondo no actúa", not d.acted and d.new_state == S)

print()
if fallos:
    print(f"{fallos} FALLA(S)")
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
