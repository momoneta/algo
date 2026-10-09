"""Reconocer objetos en pantalla a partir de ejemplos que marca el usuario (madera, piedra...).

Cada objeto se guarda como un histograma de colores de sus ejemplos. En cada fotograma
se compara cada pixel con el objeto y con el "fondo" (lo que se suele ver en el juego):
si el color es mucho mas tipico del objeto que del fondo, ese pixel es del objeto.

Solo usa numpy, para que la interfaz pueda usarlo sin cargar PyTorch.
"""
import glob
import os

import numpy as np

NIVELES = 16                      # niveles por canal de color (16*16*16 = 4096 colores)
N_COLORES = NIVELES ** 3
CARPETA = "objetos"


def indices_color(rgb):
    """(H, W, 3) uint8 -> (H, W) indice de color 0..4095."""
    q = (rgb // (256 // NIVELES)).astype(np.int32)
    return (q[..., 0] * NIVELES + q[..., 1]) * NIVELES + q[..., 2]


def histograma(rgb):
    return np.bincount(indices_color(rgb).ravel(), minlength=N_COLORES).astype(np.float64)


def ruta_objeto(juego, objeto, raiz="."):
    return os.path.join(raiz, CARPETA, juego, f"{objeto}.npy")


def anadir_ejemplo(juego, objeto, rgb, raiz=".", pantalla=None):
    """Suma un ejemplo (recorte de pantalla RGB) al objeto. Devuelve cuantos pixeles tiene ya.

    Si se da la pantalla entera, se quitan del ejemplo los colores que tambien abundan en el resto
    de la pantalla (si al marcar se cuela un trozo de hierba o de cielo, no cuenta como objeto)."""
    ruta = ruta_objeto(juego, objeto, raiz)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    h = histograma(rgb)
    if pantalla is not None:
        hp = histograma(pantalla)
        propio = h / h.sum() > 1.5 * hp / hp.sum()
        if h[propio].sum() > 0.2 * h.sum():  # si casi todo se descartaria, mejor no tocar nada
            h = h * propio
    if os.path.exists(ruta):
        h += np.load(ruta)
    np.save(ruta, h)
    return int(h.sum())


def objetos_de(juego, raiz="."):
    """{nombre: histograma} de los objetos ensenados para ese juego."""
    return {os.path.basename(p)[:-4]: np.load(p)
            for p in sorted(glob.glob(os.path.join(raiz, CARPETA, juego, "*.npy")))}


def borrar_objeto(juego, objeto, raiz="."):
    ruta = ruta_objeto(juego, objeto, raiz)
    if os.path.exists(ruta):
        os.remove(ruta)


def capturar_zona(zona):
    """Recorte RGB de la pantalla. zona = (x, y, ancho, alto)."""
    import mss
    with (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) as sct:
        img = np.asarray(sct.grab({"left": zona[0], "top": zona[1], "width": zona[2], "height": zona[3]}))
    return np.ascontiguousarray(img[:, :, :3][:, :, ::-1])


def suavizar(p):
    """Reparte cada color un poco a sus vecinos en el cubo de colores (tolerancia a luz y sombras)."""
    cubo = p.reshape(NIVELES, NIVELES, NIVELES)
    out = cubo.copy()
    for eje in range(3):
        for paso in (-1, 1):
            out += 0.3 * np.roll(cubo, paso, axis=eje)
    return out.ravel()


def reducir(rgb, tam=96):
    """Imagen pequena SIN mezclar colores (vecino mas cercano): los colores siguen siendo los reales."""
    from PIL import Image
    return np.array(Image.fromarray(rgb).resize((tam, tam), Image.NEAREST))


class Detector:
    """Detecta los objetos ensenados en cada fotograma (imagen RGB pequena)."""

    def __init__(self, objetos, umbral=0.6):
        self.nombres = list(objetos)
        tabla = []
        for h in objetos.values():
            p = h / max(h.sum(), 1)
            p[p < 0.002] = 0  # colores muy raros en los ejemplos: ruido (bordes)
            p = suavizar(p)   # tonos vecinos tambien cuentan (algo mas de luz o de sombra)
            tabla.append(p / max(p.sum(), 1e-9))
        self.p_objeto = np.array(tabla)                  # (n_objetos, 4096)
        self.p_fondo = None
        self.umbral = umbral

    def detectar(self, rgb, aprender_fondo=True):
        """rgb (H, W, 3) -> mascaras (n_objetos, H, W) bool."""
        idx = indices_color(rgb)
        h = np.bincount(idx.ravel(), minlength=N_COLORES) / idx.size
        if self.p_fondo is None:
            self.p_fondo = h.copy()
        elif aprender_fondo:
            self.p_fondo = 0.995 * self.p_fondo + 0.005 * h   # el fondo se aprende poco a poco
        po = self.p_objeto[:, idx]                       # (n, H, W)
        pf = self.p_fondo[idx][None]
        # proporcion "objeto frente a fondo": alta si ese color es tipico del objeto y raro en el fondo
        return po / (po + pf + 1e-9) > self.umbral

    @staticmethod
    def medir(mascara):
        """(visible, centro, mira): parte del objeto en toda la pantalla, en el tercio central
        y justo en la mira (el centro exacto, donde se pica o se golpea)."""
        h, w = mascara.shape
        centro = mascara[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
        mira = mascara[2 * h // 5: 3 * h // 5, 2 * w // 5: 3 * w // 5]
        return float(mascara.mean()), float(centro.mean()), float(mira.mean())


class DetectorAuto:
    """Detecta 'cosas' SIN ensenarle nada, en cualquier juego.

    Una 'cosa' es una zona compacta que destaca: su color ocupa poco de la imagen actual (no es cielo,
    suelo ni pared) o es raro en ese juego (minerales, enemigos, cofres...). Lo que nunca cambia en
    pantalla aunque se mueva la camara (barra de objetos, vida, la mira) es la interfaz y se ignora.
    """

    def __init__(self, destaca=0.04, raro=0.0015, calentamiento=40, max_pantalla=0.4):
        self.p_fondo = None
        self.n = 0
        self.destaca = destaca
        self.raro = raro
        self.calentamiento = calentamiento
        self.max_pantalla = max_pantalla
        self.anterior = None
        self.movimiento = None   # cuanto cambia cada pixel (media); ~0 = interfaz fija

    def _aprender(self, h, ritmo):
        self.p_fondo = h.copy() if self.p_fondo is None else (1 - ritmo) * self.p_fondo + ritmo * h

    def detectar(self, rgb):
        idx = indices_color(rgb)
        total = np.bincount(idx.ravel(), minlength=N_COLORES) / idx.size
        gris = rgb.mean(2)
        if self.anterior is not None:
            cambio = np.abs(gris - self.anterior)
            if cambio.mean() > 2:  # la camara se ha movido: se ve que pixeles nunca cambian
                self.movimiento = cambio if self.movimiento is None else 0.97 * self.movimiento + 0.03 * cambio
        self.anterior = gris
        self.n += 1
        self._aprender(total, max(1.0 / self.n, 0.01))
        if self.n <= self.calentamiento:  # primeros fotogramas: solo aprende como es el juego
            return np.zeros(idx.shape, bool)

        en_imagen = suavizar(total)
        en_imagen /= en_imagen.sum()
        fondo = suavizar(self.p_fondo)
        fondo /= fondo.sum()
        candidato = (en_imagen[idx] < self.destaca) | (fondo[idx] < self.raro)
        if self.movimiento is not None:
            candidato &= self.movimiento > 1.0  # fuera la interfaz fija
        # quitar puntitos sueltos: un pixel cuenta si la mayoria de sus vecinos tambien
        r = candidato.astype(np.int8)
        p = np.pad(r, 1)
        vecinos = sum(p[1 + dy:1 + dy + r.shape[0], 1 + dx:1 + dx + r.shape[1]]
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        mascara = candidato & (vecinos >= 5)
        if mascara.mean() > self.max_pantalla:  # media pantalla "destaca" = menu o cambio de escena
            return np.zeros(idx.shape, bool)
        return mascara
