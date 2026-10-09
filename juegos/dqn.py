"""Agente DQN (Deep Q-Network) con Double DQN, replay buffer y red objetivo.

Funciona con observaciones vectoriales (MLP) o imagenes (CNN, p. ej. Atari).
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dispositivo import elegir_dispositivo


class RedMLP(nn.Module):
    def __init__(self, n_entradas, n_acciones, oculta=256):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_entradas, oculta), nn.ReLU(),
            nn.Linear(oculta, oculta), nn.ReLU(),
            nn.Linear(oculta, n_acciones),
        )

    def forward(self, x):
        return self.red(x.flatten(1))


class RedCNN(nn.Module):
    """Arquitectura clasica de DQN para Atari. Entrada: (canales, 84, 84)."""

    def __init__(self, forma, n_acciones):
        super().__init__()
        c = forma[0]
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, *forma)).shape[1]
        self.cabeza = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), nn.Linear(512, n_acciones))

    def forward(self, x):
        return self.cabeza(self.conv(x / 255.0))


class ReplayBuffer:
    def __init__(self, capacidad):
        self.datos = deque(maxlen=capacidad)

    def guardar(self, obs, accion, recompensa, sig_obs, terminado):
        self.datos.append((obs, accion, recompensa, sig_obs, terminado))

    def muestra(self, n):
        lote = random.sample(self.datos, n)
        obs, acc, rec, sig, fin = zip(*lote)
        return (np.array(obs), np.array(acc), np.array(rec, dtype=np.float32),
                np.array(sig), np.array(fin, dtype=np.float32))

    def __len__(self):
        return len(self.datos)


class AgenteDQN:
    def __init__(self, forma_obs, n_acciones, lr=1e-3, gamma=0.99, buffer=100_000,
                 lote=64, actualizar_objetivo=1000, dispositivo=None):
        self.n_acciones = n_acciones
        self.gamma = gamma
        self.lote = lote
        self.actualizar_objetivo = actualizar_objetivo
        self.dispositivo = dispositivo or elegir_dispositivo()
        self.forma_obs = tuple(forma_obs)

        es_imagen = len(self.forma_obs) == 3
        crear = (lambda: RedCNN(self.forma_obs, n_acciones)) if es_imagen else \
                (lambda: RedMLP(int(np.prod(self.forma_obs)), n_acciones))
        self.red = crear().to(self.dispositivo)
        self.objetivo = crear().to(self.dispositivo)
        self.objetivo.load_state_dict(self.red.state_dict())
        self.optim = torch.optim.Adam(self.red.parameters(), lr=lr)
        self.memoria = ReplayBuffer(buffer)
        self.pasos_entreno = 0

    def actuar(self, obs, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.n_acciones)
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.dispositivo)
            return int(self.red(x.unsqueeze(0)).argmax(1).item())

    def aprender(self):
        if len(self.memoria) < self.lote:
            return None
        obs, acc, rec, sig, fin = self.memoria.muestra(self.lote)
        d = self.dispositivo
        obs = torch.as_tensor(obs, dtype=torch.float32, device=d)
        sig = torch.as_tensor(sig, dtype=torch.float32, device=d)
        acc = torch.as_tensor(acc, dtype=torch.int64, device=d)
        rec = torch.as_tensor(rec, device=d)
        fin = torch.as_tensor(fin, device=d)

        q = self.red(obs).gather(1, acc.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: la red online elige la accion, la objetivo la evalua
            mejor = self.red(sig).argmax(1, keepdim=True)
            q_sig = self.objetivo(sig).gather(1, mejor).squeeze(1)
            meta = rec + self.gamma * q_sig * (1 - fin)

        perdida = F.smooth_l1_loss(q, meta)
        self.optim.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(self.red.parameters(), 10.0)
        self.optim.step()

        self.pasos_entreno += 1
        if self.pasos_entreno % self.actualizar_objetivo == 0:
            self.objetivo.load_state_dict(self.red.state_dict())
        return perdida.item()

    def guardar(self, ruta, extra=None):
        torch.save({"red": self.red.state_dict(), "forma_obs": self.forma_obs,
                    "n_acciones": self.n_acciones, "extra": extra or {}}, ruta)

    @classmethod
    def cargar(cls, ruta, dispositivo=None):
        datos = torch.load(ruta, map_location="cpu", weights_only=False)
        agente = cls(datos["forma_obs"], datos["n_acciones"], buffer=1, dispositivo=dispositivo)
        agente.red.load_state_dict(datos["red"])
        agente.objetivo.load_state_dict(datos["red"])
        return agente, datos.get("extra", {})
