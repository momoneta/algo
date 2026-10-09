"""Utilidades para jugar a CUALQUIER juego de PC por imitacion.

Idea: grabas la pantalla y las teclas mientras juegas tu; una red neuronal
aprende "con esta imagen, el humano pulsa estas teclas"; despues la IA mira la
pantalla y pulsa las teclas ella sola.
"""
import signal
import sys
import threading

import numpy as np
import torch
import torch.nn as nn

TAM = 96        # las capturas se reducen a 96x96 en escala de grises
APILAR = 4      # la red ve los ultimos 4 fotogramas para percibir movimiento

# Nombres equivalentes de pynput -> nombre comun
ALIAS = {"shift_l": "shift", "shift_r": "shift", "ctrl_l": "ctrl", "ctrl_r": "ctrl",
         "alt_l": "alt", "alt_r": "alt", "alt_gr": "alt", "cmd_l": "cmd", "cmd_r": "cmd",
         " ": "space", "return": "enter"}


def nombre_tecla(tecla):
    """Convierte un evento de pynput en un nombre simple: 'w', 'space', 'shift', 'up'..."""
    # Primero el codigo de tecla fisica: con Ctrl o Shift pulsados, 'char' cambia
    # (Ctrl+W da '\x17', Shift+1 da '!'), pero la tecla fisica sigue siendo la misma.
    vk = getattr(tecla, "vk", None)
    if vk is not None and (0x41 <= vk <= 0x5A or 0x61 <= vk <= 0x7A or 0x30 <= vk <= 0x39):
        return chr(vk).lower()
    char = getattr(tecla, "char", None)
    if char:
        nombre = char.lower()
    else:
        nombre = getattr(tecla, "name", None) or str(tecla)
    return ALIAS.get(nombre, nombre)


def parsear_teclas(texto):
    return [ALIAS.get(t.strip().lower(), t.strip().lower()) for t in texto.split(",") if t.strip()]


def parsear_region(texto):
    """'' -> pantalla completa; 'x,y,ancho,alto' -> region."""
    if not texto or not texto.strip():
        return None
    x, y, w, h = (int(v) for v in texto.split(","))
    return {"left": x, "top": y, "width": w, "height": h}


class Capturador:
    def __init__(self, region=None, monitor=1):
        import mss
        self.sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
        self.zona = region or self.sct.monitors[monitor]

    def captura(self):
        """Devuelve un fotograma (TAM, TAM) uint8 en escala de grises."""
        from PIL import Image
        img = self.sct.grab(self.zona)
        im = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        return np.asarray(im.convert("L").resize((TAM, TAM), Image.BILINEAR), dtype=np.uint8)


class RedImitacion(nn.Module):
    """CNN: (APILAR, TAM, TAM) -> una probabilidad por tecla."""

    def __init__(self, n_teclas):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(APILAR, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, APILAR, TAM, TAM)).shape[1]
        self.cabeza = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_teclas))

    def forward(self, x):
        return self.cabeza(self.conv(x / 255.0))


class Teclado:
    """Pulsa/suelta teclas. En Windows usa pydirectinput si esta (funciona con mas juegos)."""

    def __init__(self):
        self.directo = None
        try:
            import pydirectinput
            pydirectinput.PAUSE = 0
            self.directo = pydirectinput
        except Exception:
            from pynput.keyboard import Controller
            self.ctrl = Controller()
        self.pulsadas = set()

    def _tecla_pynput(self, nombre):
        from pynput.keyboard import Key
        return getattr(Key, nombre) if len(nombre) > 1 and hasattr(Key, nombre) else nombre

    def poner(self, nombre, pulsada):
        if pulsada == (nombre in self.pulsadas):
            return
        if self.directo:
            (self.directo.keyDown if pulsada else self.directo.keyUp)(nombre)
        else:
            t = self._tecla_pynput(nombre)
            self.ctrl.press(t) if pulsada else self.ctrl.release(t)
        (self.pulsadas.add if pulsada else self.pulsadas.discard)(nombre)

    def soltar_todo(self):
        for n in list(self.pulsadas):
            self.poner(n, False)


def avisar_al_parar(estado, clave="fin"):
    """Pone estado[clave] = True si llega SIGTERM o la linea 'parar' por la entrada
    (asi la interfaz puede detenerlo sin dejar teclas pulsadas ni perder la grabacion)."""
    def senal(*_):
        estado[clave] = True
    try:
        signal.signal(signal.SIGTERM, senal)
    except ValueError:
        pass

    def leer():
        for linea in sys.stdin:
            if linea.strip().lower() == "parar":
                estado[clave] = True
                return
    threading.Thread(target=leer, daemon=True).start()
