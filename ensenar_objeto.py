"""Ensena a la IA como es un objeto del juego (madera, piedra, hierro...).

Marcas una zona de la pantalla donde se ve SOLO ese objeto y se guardan sus colores.
Cuantos mas ejemplos (distinta luz, distancia, tipo de arbol...), mejor lo reconoce.
Lo mas comodo es hacerlo desde la interfaz (boton "Ensenar objeto").

  python ensenar_objeto.py --nombre minecraft --objeto madera --zona 600,300,40,40
  python ensenar_objeto.py --nombre minecraft --listar
  python ensenar_objeto.py --nombre minecraft --objeto madera --borrar
  python ensenar_objeto.py --nombre minecraft --probar          (muestra que ve ahora en pantalla)
"""
import argparse
import time

import numpy as np

from juegos.objetos import Detector, anadir_ejemplo, borrar_objeto, capturar_zona, objetos_de, reducir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", required=True, help="nombre del juego")
    p.add_argument("--objeto", help="nombre del objeto: madera, piedra...")
    p.add_argument("--zona", help="x,y,ancho,alto de un trozo de pantalla donde solo se ve el objeto")
    p.add_argument("--espera", type=int, default=3, help="segundos antes de capturar (para cambiar al juego)")
    p.add_argument("--listar", action="store_true")
    p.add_argument("--borrar", action="store_true")
    p.add_argument("--probar", action="store_true", help="detectar los objetos en la pantalla ahora")
    p.add_argument("--region", default="", help="zona del juego para --probar")
    p.add_argument("--dispositivo", default="auto", help=argparse.SUPPRESS)  # no usa la GPU
    args = p.parse_args()

    if args.listar:
        objetos = objetos_de(args.nombre)
        if not objetos:
            print("Todavia no has ensenado ningun objeto para este juego.")
        for nombre, h in objetos.items():
            print(f"  {nombre}: {int(h.sum())} pixeles de ejemplo")
        return

    if args.borrar:
        borrar_objeto(args.nombre, args.objeto)
        print(f"Borrado: {args.objeto}")
        return

    if args.probar:
        objetos = objetos_de(args.nombre)
        if not objetos:
            raise SystemExit("No hay objetos ensenados.")
        det = Detector(objetos)
        if args.region:
            x, y, w, h = (int(v) for v in args.region.split(","))
        else:
            import mss
            with (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) as sct:
                m = sct.monitors[1]
            x, y, w, h = m["left"], m["top"], m["width"], m["height"]
        for _ in range(30):  # unos fotogramas para que aprenda como es el fondo
            rgb = reducir(capturar_zona((x, y, w, h)))
            mascaras = det.detectar(rgb)
            time.sleep(0.05)
        for nombre, mk in zip(det.nombres, mascaras):
            vis, cen, mir = Detector.medir(mk)
            print(f"  {nombre}: ocupa el {vis * 100:.1f}% de la pantalla, {cen * 100:.0f}% del centro, "
                  f"{mir * 100:.0f}% de la mira")
        return

    if not (args.objeto and args.zona):
        raise SystemExit("Indica --objeto y --zona (o usa --listar / --probar).")
    zona = tuple(int(v) for v in args.zona.split(","))
    for s in range(args.espera, 0, -1):
        print(f"Captura en {s}...", flush=True)
        time.sleep(1)
    import mss
    with (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) as sct:
        m = sct.monitors[1]
    pantalla = capturar_zona((m["left"], m["top"], m["width"], m["height"]))
    total = anadir_ejemplo(args.nombre, args.objeto, capturar_zona(zona), pantalla=pantalla)
    print(f"Ensenado '{args.objeto}' ({total} pixeles de ejemplo en total). "
          f"Anade mas ejemplos con distinta luz o distancia para que lo reconozca mejor.")


if __name__ == "__main__":
    main()
