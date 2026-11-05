"""
Política de señales de tráfico para el reto parte 2.

Módulo puro (sin I/O, sin locks, sin tiempo propio): traduce la etiqueta
detectada por la CNN + el estado actual del robot en una decisión de política.

Tabla de comportamientos del reto:
  PARE          -> detenerse completamente y permanecer detenido hasta SIGA
  SIGA          -> reanudar solo si estaba detenido por PARE; si no, continuar
  ZONA ESCOLAR  -> reducir claramente la velocidad y seguir la línea
  VELOCIDAD 60  -> aumentar velocidad si iba reducido; si no, conservar la normal
  Prioridad PARE: ZONA ESCOLAR / VELOCIDAD 60 NO reanudan un robot detenido por PARE
"""

from dataclasses import dataclass, field
from enum import Enum


class SpeedPolicy(str, Enum):
    NORMAL = "normal"
    REDUCED = "reduced"


@dataclass(frozen=True)
class PolicyDecision:
    """Decisión pura de la política de señales."""
    new_state: object          # control.State (evita import circular; se compara por valor/nombre)
    speed_policy: SpeedPolicy
    stop_until: float | None   # None = sin parada temporizada (PARE espera a SIGA)
    cover_until: float | None  # gracia de "señal tapando la línea"
    disarm: frozenset[str] = field(default_factory=frozenset)  # señales a desarmar
    acted: bool = False        # True si la decisión ejecuta una acción (PARE/SIGA)


def apply_signal(label, state, speed_policy, now, cfg):
    """
    Aplica la tabla de comportamientos del reto.

    label        : "PARE" | "SIGA" | "ZONA ESCOLAR" | "VELOCIDAD 60" | None
    state        : control.State actual del Brain
    speed_policy : SpeedPolicy actual
    now          : reloj (time.monotonic() o el del simulador)
    cfg          : módulo config con las constantes

    Devuelve PolicyDecision. La prioridad PARE está codificada como primer
    if: ZONA/VELOCIDAD solo aplican cuando state == SIGUIENDO (nunca en PARE).
    """
    cover = now + cfg.SIGNAL_COVER_GRACE_S
    sp = SpeedPolicy(speed_policy)

    state_name = getattr(state, "name", str(state))

    # PARE: prioridad absoluta, solo desde estados en movimiento
    if label == "PARE" and state_name in ("SIGUIENDO", "BUSCANDO"):
        return PolicyDecision(
            new_state=state.__class__.PARE,
            speed_policy=SpeedPolicy.NORMAL,
            stop_until=None,  # sin temporizador: espera a SIGA
            cover_until=cover,
            disarm=frozenset({"PARE"}),
            acted=True,
        )

    # SIGA: reanuda solo si estaba detenido por PARE
    if label == "SIGA":
        if state_name == "PARE":
            return PolicyDecision(
                new_state=state.__class__.SIGUIENDO,
                speed_policy=sp,
                stop_until=now,
                cover_until=cover,
                disarm=frozenset({"SIGA"}),
                acted=True,
            )
        return PolicyDecision(
            new_state=state,
            speed_policy=sp,
            stop_until=None,
            cover_until=cover,
            disarm=frozenset({"SIGA"}),
            acted=False,
        )

    # ZONA ESCOLAR: reduce velocidad, sigue la línea (nunca en PARE por prioridad)
    if label == "ZONA ESCOLAR" and state_name == "SIGUIENDO":
        return PolicyDecision(
            new_state=state,
            speed_policy=SpeedPolicy.REDUCED,
            stop_until=None,
            cover_until=cover,
            disarm=frozenset(),
            acted=True,
        )

    # VELOCIDAD 60: sube velocidad solo si iba reducido; si no, conserva la normal
    if label == "VELOCIDAD 60" and state_name == "SIGUIENDO":
        new_sp = SpeedPolicy.NORMAL if sp == SpeedPolicy.REDUCED else sp
        return PolicyDecision(
            new_state=state,
            speed_policy=new_sp,
            stop_until=None,
            cover_until=cover,
            disarm=frozenset(),
            acted=(new_sp != sp),
        )

    # fondo / None / cualquier otra etiqueta: sin acción
    return PolicyDecision(
        new_state=state,
        speed_policy=sp,
        stop_until=None,
        cover_until=cover if label is not None else None,
        disarm=frozenset(),
        acted=False,
    )