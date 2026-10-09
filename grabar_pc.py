"""Graba tu partida (pantalla + teclas) para que la IA aprenda a imitarte.

  python grabar_pc.py --nombre minecraft --teclas w,a,s,d,space,shift
  python grabar_pc.py --nombre minecraft --teclas w,a,s,d,space --raton   (tambien camara y clics)

Controles mientras graba:
  F9  = empezar / pausar la grabacion
  F10 = terminar y guardar

Consejo: juega en modo ventana y usa --region x,y,ancho,alto con la zona del juego.
Graba al menos 15-30 minutos jugando de forma consistente.
"""
import argparse
import os
import time

import numpy as np

from juegos.pc import CLICS, Capturador, OyenteRaton, avisar_al_parar, nombre_tecla, parsear_region, parsear_teclas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", required=True, help="nombre del juego (carpeta de datos)")
    p.add_argument("--teclas", default="w,a,s,d,space", help="teclas a aprender, separadas por comas")
    p.add_argument("--region", default="", help="x,y,ancho,alto (vacio = pantalla completa)")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--raton", action="store_true", help="grabar tambien el movimiento y los clics del raton")
    p.add_argument("--carpeta", default="datos_pc")
    p.add_argument("--dispositivo", default="auto", help=argparse.SUPPRESS)  # grabar no usa la GPU
    args = p.parse_args()

    from pynput import keyboard

    teclas = parsear_teclas(args.teclas)
    if args.raton:
        teclas += [c for c in CLICS.values() if c not in teclas]
    pulsadas = set()
    toques = set()  # teclas pulsadas desde el ultimo fotograma (para no perder toques rapidos)
    estado = {"grabando": False, "fin": False}

    def al_pulsar(t):
        n = nombre_tecla(t)
        if n == "f9":
            estado["grabando"] = not estado["grabando"]
            print("GRABANDO..." if estado["grabando"] else "En pausa (F9 para seguir, F10 para terminar)", flush=True)
        elif n == "f10":
            estado["fin"] = True
            return False
        pulsadas.add(n)
        toques.add(n)

    def al_soltar(t):
        pulsadas.discard(nombre_tecla(t))

    oyente = keyboard.Listener(on_press=al_pulsar, on_release=al_soltar)
    oyente.start()
    raton = None
    if args.raton:
        def al_clic(nombre, pulsado):
            if pulsado:
                pulsadas.add(nombre)
                toques.add(nombre)
            else:
                pulsadas.discard(nombre)
        raton = OyenteRaton(al_clic)
    avisar_al_parar(estado)
    cap = Capturador(parsear_region(args.region))

    print(f"Teclas a aprender: {teclas}")
    print("Pulsa F9 para empezar a grabar, F10 para terminar.", flush=True)
    fotos, etiquetas, movs = [], [], []
    periodo = 1.0 / args.fps
    ultimo_aviso = time.time()
    tiempo_grabando = 0.0
    while not estado["fin"]:
        t0 = time.time()
        if estado["grabando"]:
            fotos.append(cap.captura())
            ahora = pulsadas | toques
            toques.clear()
            etiquetas.append([k in ahora for k in teclas])
            if raton:
                movs.append(raton.tomar())
        elif raton:
            raton.tomar()  # descartar el movimiento mientras esta en pausa
            if time.time() - ultimo_aviso > 10:
                print(f"  {len(fotos)} fotogramas ({len(fotos) / args.fps / 60:.1f} min)", flush=True)
                ultimo_aviso = time.time()
        time.sleep(max(0.0, periodo - (time.time() - t0)))
        if estado["grabando"]:
            tiempo_grabando += time.time() - t0
    oyente.stop()
    if raton:
        raton.parar()

    if not fotos:
        print("No se grabo nada.")
        return
    fps_real = min(args.fps, len(fotos) / max(tiempo_grabando, 1e-6))
    if fps_real < args.fps * 0.9:
        print(f"Aviso: tu PC solo llego a {fps_real:.1f} fps (pediste {args.fps:g}). "
              f"Usa --region con la zona del juego o menos --fps.")
    carpeta = os.path.join(args.carpeta, args.nombre)
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"sesion_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    np.savez_compressed(ruta, fotos=np.stack(fotos), etiquetas=np.array(etiquetas, dtype=np.uint8),
                        teclas=np.array(teclas), fps=round(fps_real, 2), region=args.region,
                        mov=np.array(movs if raton else np.zeros((len(fotos), 2)), dtype=np.float32),
                        raton=bool(raton))
    e = np.array(etiquetas)
    print(f"Guardado {ruta}: {len(fotos)} fotogramas")
    for k, frac in zip(teclas, e.mean(0)):
        print(f"  {k:8s} pulsada el {frac * 100:5.1f}% del tiempo")
    if raton:
        m = np.abs(np.array(movs))
        print(f"  raton movido en el {(m.sum(1) > 0).mean() * 100:5.1f}% de los fotogramas "
              f"(media {m[:, 0].mean():.1f} px en X, {m[:, 1].mean():.1f} px en Y por fotograma)")


if __name__ == "__main__":
    main()
