"""Detectar el juego automaticamente: la ventana activa (la que tiene el foco), su titulo y su zona.

Solo usa la biblioteca estandar (y python-xlib en Linux), para que la interfaz pueda usarlo
sin cargar PyTorch.
"""
import re
import sys


def ventana_activa():
    """(titulo, (x, y, ancho, alto)) del area de juego de la ventana activa, o (None, None)."""
    try:
        if sys.platform == "win32":
            return _windows()
        return _linux()
    except Exception:
        return None, None


def _windows():
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # coordenadas reales en pantallas escaladas
    except Exception:
        pass
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None, None
    largo = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(largo + 1)
    user32.GetWindowTextW(hwnd, buf, largo + 1)
    # area "cliente": sin bordes ni barra de titulo (solo lo que dibuja el juego)
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    punto = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(punto))
    return buf.value, (punto.x, punto.y, rect.right - rect.left, rect.bottom - rect.top)


def _linux():
    from Xlib import X, display
    d = display.Display()
    raiz = d.screen().root
    activa = raiz.get_full_property(d.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType)
    if activa and activa.value and activa.value[0]:
        ventana = d.create_resource_object("window", activa.value[0])
    else:  # sin gestor de ventanas: la ventana con el foco
        ventana = d.get_input_focus().focus
        if isinstance(ventana, int):
            return None, None
    titulo = None
    w = ventana
    while w is not None and not titulo:
        nombre = w.get_full_property(d.intern_atom("_NET_WM_NAME"), 0) or w.get_full_property(X.XA_WM_NAME, 0)
        titulo = nombre.value.decode("utf-8", "replace") if nombre and isinstance(nombre.value, bytes) else \
            (nombre.value if nombre else None)
        arbol = w.query_tree()
        w = arbol.parent if arbol.parent and arbol.parent != raiz else None
    geo = ventana.get_geometry()
    pos = ventana.translate_coords(raiz, 0, 0)
    return titulo, (-pos.x, -pos.y, geo.width, geo.height)


def nombre_juego(titulo):
    """'Minecraft* 1.20.1 - Multijugador' -> 'Minecraft'. Nombre corto y valido como carpeta."""
    if not titulo:
        return "juego"
    t = re.split(r"\s*[-|:]\s|\s\d", titulo)[0]        # quita versiones y subtitulos
    t = re.sub(r"[^\w ]", "", t).strip()                 # sin simbolos raros
    t = re.sub(r"\s+", "_", t)
    return t[:40] or "juego"


def region_texto(zona):
    return ",".join(str(int(v)) for v in zona)
