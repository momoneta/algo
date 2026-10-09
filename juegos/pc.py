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

# Raton: los clics se tratan como "teclas" con estos nombres
CLICS = {"left": "clic_izq", "right": "clic_der"}
# Movimiento del raton por fotograma (pixeles), en grupos: la red elige el grupo.
# Escala "logaritmica": mucho detalle en movimientos finos (apuntar) y tambien giros rapidos.
CENTROS_MOV = np.array([-160, -100, -60, -35, -20, -10, -5, -2, 0, 2, 5, 10, 20, 35, 60, 100, 160],
                       dtype=np.float32)
CENTROS_MOV_ANTIGUOS = np.array([-80, -40, -20, -10, -4, 0, 4, 10, 20, 40, 80], dtype=np.float32)

# Teclas que la IA nunca pulsa (controles del programa, menu de Windows, Alt+F4, Esc abre menus)
PELIGROSAS = {"f9", "f10", "cmd", "f4", "esc", "print_screen", "caps_lock", "num_lock", "scroll_lock",
              "menu", "pause", "insert"}
# Teclas tipicas de juego, para aprender sola cuando no hay grabaciones de las que sacarlas
TECLAS_JUEGO = ["w", "a", "s", "d", "space", "shift", "ctrl", "e", "q", "r", "f", "c", "x", "z",
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "up", "down", "left", "right"]


def decidir_mov(logits, centros):
    """Movimiento a partir de las probabilidades de cada grupo.

    La media de TODOS los grupos tiende a dejar el raton casi quieto (izquierda y derecha se anulan),
    asi que se toma el grupo mas probable y se afina con sus dos vecinos.
    logits: (B, n_grupos)  centros: tensor (n_grupos,)  ->  (B,)
    """
    prob = torch.softmax(logits, 1)
    mejor = prob.argmax(1)
    n = prob.shape[1]
    vecinos = torch.stack([(mejor - 1).clamp(0, n - 1), mejor, (mejor + 1).clamp(0, n - 1)], 1)
    p = prob.gather(1, vecinos)
    return (p * centros[vecinos]).sum(1) / p.sum(1)


def objetivo_suave(clases, n, ancho=0.8):
    """Etiqueta 'suave': el grupo correcto y un poco sus vecinos (fallar por poco cuesta menos)."""
    i = torch.arange(n, device=clases.device, dtype=torch.float32)
    t = torch.exp(-((i[None, :] - clases[:, None].float()) ** 2) / (2 * ancho ** 2))
    return t / t.sum(1, keepdim=True)


def mov_a_clase(v):
    """Pixeles movidos -> indice del grupo mas cercano."""
    return int(np.abs(CENTROS_MOV - v).argmin())


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
    """'w,a,s,d' -> lista. 'todas' / 'auto' / '' -> None (se deciden solas)."""
    if not texto or texto.strip().lower() in ("todas", "auto", "all"):
        return None
    return [ALIAS.get(t.strip().lower(), t.strip().lower()) for t in texto.split(",") if t.strip()]


def ordenar_teclas(teclas):
    """Orden estable y legible: letras, numeros, especiales y al final los clics."""
    def clave(k):
        return (k.startswith("clic_"), len(k) > 1, k)
    return sorted(set(teclas), key=clave)


def teclas_grabadas(carpeta):
    """Todas las teclas usadas en las grabaciones de una carpeta (sin los clics)."""
    import glob
    import os
    vistas = set()
    for a in glob.glob(os.path.join(carpeta, "*.npz")):
        with np.load(a) as d:
            e = d["etiquetas"]
            for k, n in zip(d["teclas"], e.sum(0)):
                if n > 0:
                    vistas.add(str(k))
    return [k for k in ordenar_teclas(vistas) if not k.startswith("clic_") and k not in PELIGROSAS]


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
        return np.array(im.convert("L").resize((TAM, TAM), Image.BILINEAR), dtype=np.uint8)

    def fraccion_color(self, zona, color, tolerancia=60):
        """Que parte de la zona tiene un color parecido a 'color' (0..1). Sirve para leer barras."""
        img = np.asarray(self.sct.grab(zona), dtype=np.int16)[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        dist = np.abs(img - np.array(color, dtype=np.int16)).sum(2)
        return float((dist < tolerancia).mean())


class RedImitacion(nn.Module):
    """CNN: (APILAR, TAM, TAM) -> una probabilidad por tecla (+ movimiento del raton).

    Salida: [n_teclas logits | len(CENTROS_MOV) logits para X | len(CENTROS_MOV) logits para Y]
    """

    def __init__(self, n_teclas, raton=False, n_mov=len(CENTROS_MOV)):
        super().__init__()
        self.n_teclas = n_teclas
        self.raton = raton
        self.n_mov = n_mov
        n_salida = n_teclas + (2 * n_mov if raton else 0)
        self.conv = nn.Sequential(
            nn.Conv2d(APILAR, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, APILAR, TAM, TAM)).shape[1]
        self.cabeza = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_salida))

    def forward(self, x):
        """Devuelve (logits_teclas, logits_x, logits_y); los de raton son None si no hay raton."""
        out = self.cabeza(self.conv(x / 255.0))
        teclas = out[:, :self.n_teclas]
        if not self.raton:
            return teclas, None, None
        nb = self.n_mov
        return teclas, out[:, self.n_teclas:self.n_teclas + nb], out[:, self.n_teclas + nb:]


class OyenteRaton:
    """Acumula cuanto se mueve el raton entre fotogramas y avisa de los clics.

    Funciona tambien en juegos que bloquean el cursor en el centro (Minecraft, FPS):
    en Windows se compara cada movimiento con la posicion real del cursor en ese momento.
    """

    def __init__(self, al_clic):
        from pynput import mouse
        self.dx = self.dy = 0.0
        self.ultimo = None
        self.cerrojo = threading.Lock()
        self._cursor = None
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            punto = wintypes.POINT()

            def cursor():
                ctypes.windll.user32.GetCursorPos(ctypes.byref(punto))
                return punto.x, punto.y
            self._cursor = cursor

        def al_mover(x, y):
            previo = self._cursor() if self._cursor else self.ultimo
            self.ultimo = (x, y)
            if previo is None:
                return
            with self.cerrojo:
                self.dx += x - previo[0]
                self.dy += y - previo[1]

        def clic(x, y, boton, pulsado):
            nombre = CLICS.get(getattr(boton, "name", ""))
            if nombre:
                al_clic(nombre, pulsado)

        self.oyente = mouse.Listener(on_move=al_mover, on_click=clic)
        self.oyente.start()

    def tomar(self):
        """Movimiento acumulado desde la ultima llamada."""
        with self.cerrojo:
            d = (self.dx, self.dy)
            self.dx = self.dy = 0.0
        return d

    def parar(self):
        self.oyente.stop()


class Teclado:
    """Pulsa/suelta teclas. En Windows usa pydirectinput si esta (funciona con mas juegos)."""

    def __init__(self):
        self.directo = None
        try:
            import pydirectinput
            pydirectinput.PAUSE = 0
            pydirectinput.FAILSAFE = False
            self.directo = pydirectinput
        except Exception:
            from pynput.keyboard import Controller
            from pynput.mouse import Controller as ControllerRaton
            self.ctrl = Controller()
            self.raton = ControllerRaton()
        self.pulsadas = set()

    def mover(self, dx, dy):
        dx, dy = int(round(dx)), int(round(dy))
        if not dx and not dy:
            return
        if self.directo:
            # movimiento relativo real (SendInput): los juegos 3D lo detectan
            self.directo.moveRel(dx, dy, relative=True)
        else:
            self.raton.move(dx, dy)

    def mover_suave(self, dx, dy, tiempo, pasos=6):
        """Reparte el movimiento en varios pasos pequenos durante 'tiempo' segundos (camara fluida)."""
        import time
        pasos = max(1, pasos)
        hecho_x = hecho_y = 0
        for i in range(1, pasos + 1):
            # se redondea el acumulado para no perder decimales por el camino
            ox, oy = round(dx * i / pasos), round(dy * i / pasos)
            self.mover(ox - hecho_x, oy - hecho_y)
            hecho_x, hecho_y = ox, oy
            if tiempo > 0:
                time.sleep(tiempo / pasos)

    def _clic(self, nombre, pulsado):
        boton = "left" if nombre == "clic_izq" else "right"
        if self.directo:
            (self.directo.mouseDown if pulsado else self.directo.mouseUp)(button=boton)
        else:
            from pynput.mouse import Button
            b = getattr(Button, boton)
            self.raton.press(b) if pulsado else self.raton.release(b)

    def _tecla_pynput(self, nombre):
        from pynput.keyboard import Key
        return getattr(Key, nombre) if len(nombre) > 1 and hasattr(Key, nombre) else nombre

    def poner(self, nombre, pulsada):
        if pulsada == (nombre in self.pulsadas):
            return
        if nombre in CLICS.values():
            self._clic(nombre, pulsada)
        elif self.directo:
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
