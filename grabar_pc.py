"""Graba tu partida (pantalla + teclas) para que la IA aprenda a imitarte.

  python grabar_pc.py --nombre minecraft --teclas w,a,s,d,space,shift

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

from juegos.pc import Capturador, avisar_al_parar, nombre_tecla, parsear_region, parsear_teclas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", required=True, help="nombre del juego (carpeta de datos)")
    p.add_argument("--teclas", default="w,a,s,d,space", help="teclas a aprender, separadas por comas")
    p.add_argument("--region", default="", help="x,y,ancho,alto (vacio = pantalla completa)")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--carpeta", default="datos_pc")
    args = p.parse_args()

    from pynput import keyboard

    teclas = parsear_teclas(args.teclas)
    pulsadas = set()
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

    def al_soltar(t):
        pulsadas.discard(nombre_tecla(t))

    oyente = keyboard.Listener(on_press=al_pulsar, on_release=al_soltar)
    oyente.start()
    avisar_al_parar(estado)
    cap = Capturador(parsear_region(args.region))

    print(f"Teclas a aprender: {teclas}")
    print("Pulsa F9 para empezar a grabar, F10 para terminar.", flush=True)
    fotos, etiquetas = [], []
    periodo = 1.0 / args.fps
    ultimo_aviso = time.time()
    while not estado["fin"]:
        t0 = time.time()
        if estado["grabando"]:
            fotos.append(cap.captura())
            etiquetas.append([k in pulsadas for k in teclas])
            if time.time() - ultimo_aviso > 10:
                print(f"  {len(fotos)} fotogramas ({len(fotos) / args.fps / 60:.1f} min)", flush=True)
                ultimo_aviso = time.time()
        time.sleep(max(0.0, periodo - (time.time() - t0)))
    oyente.stop()

    if not fotos:
        print("No se grabo nada.")
        return
    carpeta = os.path.join(args.carpeta, args.nombre)
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"sesion_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    np.savez_compressed(ruta, fotos=np.stack(fotos), etiquetas=np.array(etiquetas, dtype=np.uint8),
                        teclas=np.array(teclas), fps=args.fps, region=args.region)
    e = np.array(etiquetas)
    print(f"Guardado {ruta}: {len(fotos)} fotogramas")
    for k, frac in zip(teclas, e.mean(0)):
        print(f"  {k:8s} pulsada el {frac * 100:5.1f}% del tiempo")


if __name__ == "__main__":
    main()
