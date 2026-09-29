"""
Memoria del camino: odometría + mapa de cruces + exploración en profundidad.

El robot no tiene encoders, así que la posición se estima integrando los
mismos comandos que se envían (modelo uniciclo, igual que sim.py):
    x += v·cos(θ)·dt      y += v·sin(θ)·dt      θ += ω·dt
Es "navegación a estima": acumula error, por eso cada vez que se reconoce
un cruce ya visitado se corrige la posición (cierre de lazo).

Mapa topológico (un grafo):
  * Nodo  = una ramificación, con su posición (x, y) en cm.
  * Rama  = cada camino que sale del nodo, identificado por su RUMBO absoluto
            (grados, antihorario desde +x) y un estado:
              pendiente   → vista pero no recorrida
              explorando  → es por donde vamos ahora
              sin_salida  → terminó en callejón (o todo lo que hay detrás lo es)
              origen      → por donde se llegó al nodo la primera vez

Estrategia (búsqueda en profundidad, DFS):
  1. Cruce nuevo → se guarda con sus ramas y se toma una pendiente
     (BRANCH_POLICY). El nodo se apila.
  2. Callejón → la rama actual queda sin_salida, media vuelta y modo
     REGRESANDO hasta el nodo de la cima de la pila.
  3. De vuelta en el nodo → si queda alguna pendiente se toma (EXPLORANDO);
     si no, el nodo está agotado: se desapila, la rama que llevaba a él se
     marca sin_salida en el nodo anterior y se sale por la rama origen.
  4. Pila vacía y otro callejón → no hay más caminos: FIN.

Todo se llama con el candado de Brain tomado; aquí no hay hilos propios.
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

import config

PENDIENTE, EXPLORANDO, SIN_SALIDA, ORIGEN = "pendiente", "explorando", "sin_salida", "origen"


def ang_diff(a, b):
    """Diferencia a − b en grados, llevada a (−180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


@dataclass
class Branch:
    heading: float                  # Rumbo absoluto de la rama (grados)
    status: str = PENDIENTE


@dataclass
class Node:
    id: int
    pos: np.ndarray                 # (x, y) en cm, marco de la odometría
    branches: list = field(default_factory=list)

    def pending(self):
        return [b for b in self.branches if b.status == PENDIENTE]


class Odometry:
    """Pose (x, y, θ) estimada a partir de los comandos enviados."""

    def __init__(self):
        self.x = self.y = 0.0
        self.theta = 90.0           # Grados; arranca "mirando hacia arriba" en el mapa
        self.dist = 0.0             # cm recorridos en total
        self.turned = 0.0           # Grados girados en total (con signo)

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def update(self, v, w_deg, dt):
        """Integra (v cm/s, ω °/s) durante dt en sub-pasos de 10 ms."""
        steps = max(1, int(dt / 0.01))
        h = dt / steps
        for _ in range(steps):
            t = math.radians(self.theta)
            self.x += v * math.cos(t) * h
            self.y += v * math.sin(t) * h
            self.theta += w_deg * h
        self.dist += abs(v) * dt
        self.turned += w_deg * dt

    def to_world(self, fwd, right):
        """Punto (adelante, derecha) en cm relativo al robot → (x, y) del mapa."""
        t = math.radians(self.theta)
        # Vector adelante = (cosθ, sinθ); derecha = adelante rotado −90° = (sinθ, −cosθ)
        return np.array([self.x + fwd * math.cos(t) + right * math.sin(t),
                         self.y + fwd * math.sin(t) - right * math.cos(t)])


class Navigator:
    """Decide qué rama tomar en cada cruce y recuerda el camino recorrido."""

    def __init__(self):
        self.odom = Odometry()
        self.nodes = []
        self.stack = []             # Nodos del camino actual, del inicio a la cima
        self.mode = "EXPLORANDO"    # EXPLORANDO / REGRESANDO / FIN
        self.commit = None          # (nodo, rumbo, dist_al_decidir) mientras se toma una rama
        self._commit_frames = 0
        self._returning_on = None   # Rama por la que se está regresando (para corregir el rumbo)
        self.trail = [(0.0, 0.0)]
        self.events = []            # Últimos eventos, para mostrar en pantalla

    # ── Odometría ──────────────────────────────────────────────────────────
    def on_motion(self, v, w_deg, dt):
        if not (v or w_deg):
            return
        self.odom.update(v, w_deg, dt)
        if math.hypot(self.odom.x - self.trail[-1][0], self.odom.y - self.trail[-1][1]) > 1.0:
            self.trail.append((self.odom.x, self.odom.y))
        # La rama elegida se mantiene COMMIT_CM; después vuelve el seguimiento normal
        if self.commit is not None and self.odom.dist - self.commit[2] > config.COMMIT_CM:
            self._set_commit(None)

    # ── Geometría de las salidas ───────────────────────────────────────────
    def _exit_headings(self, line):
        """
        Rumbo absoluto de cada salida. El centro del cruce J se estima como el
        promedio de la entrada y las salidas (en cm sobre el piso); el rumbo
        de la rama es la dirección J → salida, girada por θ del robot:
            rumbo = θ − atan2(derecha, adelante)
        (− porque "derecha" es sentido horario y θ es antihorario).
        """
        pts = np.array([e.floor for e in line.exits] + [line.entry_floor])
        j = pts.mean(axis=0)
        heads = [self.odom.theta - math.degrees(math.atan2(e.floor[1] - j[1], e.floor[0] - j[0]))
                 for e in line.exits]
        return heads, self.odom.to_world(j[0], j[1])

    def _log(self, text):
        print(f"[MAPA] {text}")
        self.events = (self.events + [text])[-4:]

    # ── Eventos ────────────────────────────────────────────────────────────
    def on_junction(self, line):
        """Se confirmó una ramificación a la vista. Decide la rama y la fija."""
        heads, center = self._exit_headings(line)
        node = self._match_node(center)

        if node is None:
            # Cruce nuevo: una rama por salida + la rama por donde se llegó
            node = Node(len(self.nodes), center)
            node.branches = [Branch(h % 360.0) for h in heads]
            node.branches.append(Branch((self.odom.theta + 180.0) % 360.0, ORIGEN))
            self.nodes.append(node)
            self.stack.append(node)
            self.mode = "EXPLORANDO"
            self._log(f"Cruce nuevo N{node.id} con {len(heads)} ramas")
        else:
            self._relocalize(node, line)
            heads, center = self._exit_headings(line)
            self._merge_exits(node, heads)
            if self.mode == "EXPLORANDO" and self.stack and node is not self.stack[-1]:
                # Se llegó por OTRO camino a un cruce ya visto: hay un lazo.
                # Regla de Trémaux: la rama actual no aporta nada nuevo, se
                # cierra y se da la vuelta hasta el nodo de la cima de la pila.
                self._mark_current(SIN_SALIDA)
                self.mode = "REGRESANDO"
                self._log(f"Lazo: N{node.id} ya visitado → regreso a N{self.stack[-1].id}")
                self._set_commit((node, (self.odom.theta + 180.0) % 360.0, self.odom.dist))
                return
            self._log(f"De vuelta en N{node.id}")

        choice = self._choose(node)
        if choice is None:
            # Nodo agotado: salir por el origen y marcar la rama que traía aquí
            origin = next(b for b in node.branches if b.status == ORIGEN)
            if self.stack and self.stack[-1] is node:
                self.stack.pop()
            self._mark_current(SIN_SALIDA)
            self.mode = "REGRESANDO"
            self._returning_on = self._current_branch_from(self.stack[-1]) if self.stack else None
            self._log(f"N{node.id} agotado → regreso por el origen")
            heading = origin.heading
        else:
            for b in node.branches:
                if b.status == EXPLORANDO:
                    b.status = SIN_SALIDA if b is not choice else EXPLORANDO
            choice.status = EXPLORANDO
            self.mode = "EXPLORANDO"
            self._returning_on = None
            rel = ang_diff(choice.heading, self.odom.theta)
            self._log(f"N{node.id}: tomo rama a {'izq' if rel > 0 else 'der'} {abs(rel):.0f}°")
            heading = choice.heading
        self._set_commit((node, heading, self.odom.dist))

    def on_dead_end(self):
        """La línea terminó. Devuelve False si ya no queda nada por explorar."""
        self._set_commit(None)
        if not self.stack:
            self.mode = "FIN"
            self._log("Sin caminos pendientes: FIN")
            return False
        self._returning_on = self._current_branch_from(self.stack[-1])
        self._mark_current(SIN_SALIDA)
        self.mode = "REGRESANDO"
        self._log(f"Callejón sin salida → regreso a N{self.stack[-1].id}")
        return True

    # ── Control mientras se toma una rama ──────────────────────────────────
    def steer_error(self, line):
        """
        Error de control mientras hay una rama fijada, o None si no la hay.
          * Si se ve una salida con rumbo parecido al elegido → apuntar a ella
            (persecución pura: error = ángulo hacia el punto / STEER_FULL_DEG).
          * Si no se ve y el cruce todavía está delante, o ya se va casi en el
            rumbo elegido → seguir la línea.
          * Si no → girar hasta el rumbo elegido (p. ej. rama muy cerrada).
        """
        if self.commit is None:
            return None
        self._commit_frames += 1
        if self._commit_frames > config.COMMIT_MAX_FRAMES:
            self._log("Rama fijada demasiado tiempo → sigo la línea")
            self._set_commit(None)
            return None
        node, heading, _ = self.commit
        if line is not None and line.exits:
            heads, _ = self._exit_headings(line)
            diffs = [abs(ang_diff(h, heading)) for h in heads]
            i = int(np.argmin(diffs))
            if diffs[i] < config.BRANCH_MATCH_DEG:
                return float(np.clip(line.exits[i].angle / config.STEER_FULL_DEG, -1, 1))
        turn = ang_diff(heading, self.odom.theta)       # > 0 = la rama queda a la izquierda
        t = math.radians(self.odom.theta)
        ahead = np.dot(node.pos - self.odom.pos, [math.cos(t), math.sin(t)])
        # Seguir la línea si el cruce aún está delante o si ya se va más o menos
        # en el rumbo elegido (el rumbo de la odometría es aproximado: la línea manda)
        if line is not None and (ahead > config.NODE_REACH_CM
                                 or abs(turn) < config.BRANCH_MATCH_DEG):
            return line.error
        # Error de rumbo: > 0 = hay que girar a la derecha (θ debe bajar)
        return float(np.clip(-turn / config.STEER_FULL_DEG, -1, 1))

    # ── Auxiliares ─────────────────────────────────────────────────────────
    def _set_commit(self, commit):
        self.commit = commit
        self._commit_frames = 0

    def _relocalize(self, node, line):
        """
        Cierre de lazo al reconocer un cruce ya visitado. Sin encoders el
        rumbo θ es lo que más deriva (cada media vuelta suma error), así que:
          1. Si se regresa por una rama conocida, el robot la recorre al revés:
             θ ≈ rumbo_rama + 180°. Se corrige una fracción HEADING_FIX_GAIN.
          2. Se emparejan las salidas vistas con las ramas guardadas y θ se
             ajusta con el promedio de las diferencias (solo las < tolerancia).
          3. La posición se lleva a la del nodo guardado.
        """
        if self._returning_on is not None:
            expected = self._returning_on.heading + 180.0
            self.odom.theta += config.HEADING_FIX_GAIN * ang_diff(expected, self.odom.theta)
            self._returning_on = None
        heads, _ = self._exit_headings(line)
        diffs = []
        for h in heads:
            d = min((ang_diff(b.heading, h) for b in node.branches), key=abs)
            if abs(d) < config.BRANCH_MATCH_DEG:
                diffs.append(d)
        if diffs:
            self.odom.theta += config.HEADING_FIX_GAIN * float(np.mean(diffs))
        _, center = self._exit_headings(line)
        self.odom.x, self.odom.y = self.odom.pos + (node.pos - center)

    def _match_node(self, center):
        """Nodo conocido a menos de NODE_MATCH_CM; al regresar, la cima de la pila."""
        best = min(self.nodes, key=lambda n: np.linalg.norm(n.pos - center), default=None)
        if best is not None and np.linalg.norm(best.pos - center) < config.NODE_MATCH_CM:
            return best
        if self.mode == "REGRESANDO" and self.stack:
            return self.stack[-1]   # La odometría derivó, pero solo puede ser este
        return None

    def _merge_exits(self, node, heads):
        """Ramas que se ven ahora y no estaban en el nodo → se agregan (el mapa crece)."""
        for h in heads:
            if all(abs(ang_diff(h, b.heading)) > config.BRANCH_MATCH_DEG for b in node.branches):
                node.branches.append(Branch(h % 360.0))

    def _choose(self, node):
        """Rama pendiente según BRANCH_POLICY (ángulo relativo: > 0 = izquierda)."""
        pend = node.pending()
        if not pend:
            return None
        rel = lambda b: ang_diff(b.heading, self.odom.theta)
        if config.BRANCH_POLICY == "derecha":
            return min(pend, key=rel)
        if config.BRANCH_POLICY == "recto":
            return min(pend, key=lambda b: abs(rel(b)))
        return max(pend, key=rel)

    def _current_branch_from(self, node):
        return next((b for b in node.branches if b.status == EXPLORANDO), None)

    def _mark_current(self, status):
        """Marca la rama que se está recorriendo desde la cima de la pila."""
        if self.stack:
            b = self._current_branch_from(self.stack[-1])
            if b is not None:
                b.status = status

    # ── Dibujo ─────────────────────────────────────────────────────────────
    def snapshot(self):
        """Copia de lo necesario para dibujar (se llama con el candado de Brain)."""
        return {"trail": list(self.trail), "pose": (self.odom.x, self.odom.y, self.odom.theta),
                "nodes": [(n.id, n.pos.copy(), [(b.heading, b.status) for b in n.branches])
                          for n in self.nodes],
                "stack": [n.id for n in self.stack], "mode": self.mode,
                "events": list(self.events)}


BRANCH_COLORS = {PENDIENTE: (255, 255, 255), EXPLORANDO: (0, 220, 0),
                 SIN_SALIDA: (0, 0, 255), ORIGEN: (255, 160, 0)}


def draw_map(snap, size=420):
    """Ventana "Mapa (odometria)": rastro, cruces con sus ramas y el robot."""
    img = np.full((size, size, 3), 25, np.uint8)
    pts = np.array(snap["trail"] + [snap["pose"][:2]] + [n[1] for n in snap["nodes"]])
    lo, hi = pts.min(axis=0) - 20, pts.max(axis=0) + 20
    k = (size - 40) / max(hi[0] - lo[0], hi[1] - lo[1], 60.0)       # px por cm, todo cabe
    to_px = lambda p: (int(20 + (p[0] - lo[0]) * k), int(size - 20 - (p[1] - lo[1]) * k))

    if len(snap["trail"]) > 1:
        cv2.polylines(img, [np.array([to_px(p) for p in snap["trail"]], np.int32)],
                      False, (0, 200, 255), 1, cv2.LINE_AA)
    for nid, pos, branches in snap["nodes"]:
        c = to_px(pos)
        for heading, status in branches:
            t = math.radians(heading)
            end = to_px(pos + 12 * np.array([math.cos(t), math.sin(t)]))
            cv2.line(img, c, end, BRANCH_COLORS[status], 2, cv2.LINE_AA)
        in_stack = nid in snap["stack"]
        cv2.circle(img, c, 6, (0, 255, 255) if in_stack else (120, 120, 120), -1)
        cv2.putText(img, f"N{nid}", (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

    x, y, th = snap["pose"]
    t = math.radians(th)
    fwd, left = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    p = np.array([x, y])
    tri = [to_px(p + fwd * 8), to_px(p - fwd * 6 + left * 5), to_px(p - fwd * 6 - left * 5)]
    cv2.fillPoly(img, [np.array(tri, np.int32)], (0, 255, 0))

    cv2.putText(img, f"{snap['mode']}  pila={snap['stack']}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, ev in enumerate(snap["events"]):
        cv2.putText(img, ev, (8, 38 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 180, 180), 1, cv2.LINE_AA)
    return img
