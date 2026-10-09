"""Juego Snake sencillo con la misma API que Gymnasium (reset/step).

No necesita librerias externas aparte de numpy, asi que sirve para probar
el entrenamiento sin instalar nada mas.
"""
import numpy as np

# Direcciones: arriba, derecha, abajo, izquierda (en sentido horario)
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class Discrete:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(self.n)


class Box:
    def __init__(self, shape):
        self.shape = shape


class SnakeEnv:
    """Acciones: 0 = seguir recto, 1 = girar a la derecha, 2 = girar a la izquierda.

    Observacion (11 valores):
      peligro recto/derecha/izquierda, direccion actual (4),
      comida a la izquierda/derecha/arriba/abajo.
    """

    def __init__(self, ancho=10, alto=10, max_pasos_sin_comer=100, render_mode=None):
        self.ancho = ancho
        self.alto = alto
        self.max_pasos_sin_comer = max_pasos_sin_comer
        self.render_mode = render_mode
        self.action_space = Discrete(3)
        self.observation_space = Box((11,))
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cx, cy = self.ancho // 2, self.alto // 2
        self.serpiente = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direccion = 1  # derecha
        self.puntos = 0
        self.pasos_sin_comer = 0
        self._poner_comida()
        return self._obs(), {}

    def _poner_comida(self):
        libres = [(x, y) for x in range(self.ancho) for y in range(self.alto)
                  if (x, y) not in self.serpiente]
        self.comida = libres[self.rng.integers(len(libres))] if libres else None

    def _choca(self, punto):
        x, y = punto
        return (x < 0 or x >= self.ancho or y < 0 or y >= self.alto
                or punto in self.serpiente[:-1])

    def _obs(self):
        cabeza = self.serpiente[0]
        d = self.direccion

        def delante(dir_idx):
            dx, dy = DIRS[dir_idx]
            return (cabeza[0] + dx, cabeza[1] + dy)

        fx, fy = self.comida if self.comida else cabeza
        obs = [
            self._choca(delante(d)),
            self._choca(delante((d + 1) % 4)),
            self._choca(delante((d - 1) % 4)),
            d == 0, d == 1, d == 2, d == 3,
            fx < cabeza[0], fx > cabeza[0], fy < cabeza[1], fy > cabeza[1],
        ]
        return np.array(obs, dtype=np.float32)

    def step(self, accion):
        if accion == 1:
            self.direccion = (self.direccion + 1) % 4
        elif accion == 2:
            self.direccion = (self.direccion - 1) % 4

        dx, dy = DIRS[self.direccion]
        cabeza = self.serpiente[0]
        nueva = (cabeza[0] + dx, cabeza[1] + dy)

        # Distancia a la comida antes/despues, para dar una pequena pista
        dist_antes = abs(cabeza[0] - self.comida[0]) + abs(cabeza[1] - self.comida[1])

        if self._choca(nueva):
            return self._obs(), -10.0, True, False, {"puntos": self.puntos}

        self.serpiente.insert(0, nueva)
        self.pasos_sin_comer += 1

        if nueva == self.comida:
            self.puntos += 1
            self.pasos_sin_comer = 0
            recompensa = 10.0
            self._poner_comida()
            if self.comida is None:  # ha llenado el tablero
                return self._obs(), 100.0, True, False, {"puntos": self.puntos}
        else:
            self.serpiente.pop()
            dist_despues = abs(nueva[0] - self.comida[0]) + abs(nueva[1] - self.comida[1])
            recompensa = 0.1 if dist_despues < dist_antes else -0.15

        truncado = self.pasos_sin_comer > self.max_pasos_sin_comer * len(self.serpiente)
        return self._obs(), recompensa, False, truncado, {"puntos": self.puntos}

    def render(self):
        filas = []
        for y in range(self.alto):
            fila = ""
            for x in range(self.ancho):
                if (x, y) == self.serpiente[0]:
                    fila += "@"
                elif (x, y) in self.serpiente:
                    fila += "o"
                elif (x, y) == self.comida:
                    fila += "*"
                else:
                    fila += "."
            filas.append(fila)
        borde = "+" + "-" * self.ancho + "+"
        texto = "\n".join([borde] + ["|" + f + "|" for f in filas] + [borde])
        return texto + f"\nPuntos: {self.puntos}"

    def close(self):
        pass
