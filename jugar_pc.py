"""La IA juega sola: mira la pantalla y pulsa las teclas que aprendio.

  python jugar_pc.py --modelo modelos/pc_minecraft.pt

Tienes unos segundos para hacer clic en la ventana del juego.
F10 = parar (siempre suelta todas las teclas al terminar).
"""
import argparse
import time
from collections import deque

import numpy as np
import torch

from dispositivo import elegir_dispositivo, describir
from juegos.pc import APILAR, CENTROS_MOV, Capturador, RedImitacion, Teclado, avisar_al_parar, nombre_tecla, parsear_region


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True)
    p.add_argument("--umbral", type=float, default=0.5, help="probabilidad minima para pulsar una tecla")
    p.add_argument("--region", default=None, help="por defecto la misma que al grabar")
    p.add_argument("--espera", type=int, default=5, help="segundos antes de empezar")
    p.add_argument("--raton-escala", type=float, default=1.0,
                   help="multiplica el movimiento del raton (sube si gira poco, baja si gira demasiado)")
    p.add_argument("--sin-raton", action="store_true", help="no mover el raton aunque el modelo sepa")
    p.add_argument("--dispositivo", default="auto")
    args = p.parse_args()

    from pynput import keyboard

    datos = torch.load(args.modelo, map_location="cpu", weights_only=False)
    teclas = datos["teclas"]
    d = elegir_dispositivo(args.dispositivo)
    raton = datos.get("raton", False)
    modelo = RedImitacion(len(teclas), raton)
    modelo.load_state_dict(datos["modelo"])
    modelo.to(d).eval()
    region = args.region if args.region is not None else datos.get("region", "")
    fps = datos.get("fps", 10)
    usar_raton = raton and not args.sin_raton
    print(f"Dispositivo: {describir(d)} | teclas {teclas} | {fps} fps"
          + (" | mueve el raton" if usar_raton else ""))
    centros = torch.as_tensor(CENTROS_MOV, device=d)

    parar = {"si": False}

    def al_pulsar(t):
        if nombre_tecla(t) == "f10":
            parar["si"] = True
            return False

    oyente = keyboard.Listener(on_press=al_pulsar)
    oyente.start()
    avisar_al_parar(parar, "si")

    for s in range(args.espera, 0, -1):
        if parar["si"]:
            return
        print(f"Empieza en {s}... (haz clic en la ventana del juego)", flush=True)
        time.sleep(1)
    print("JUGANDO. Pulsa F10 para parar.", flush=True)

    cap = Capturador(parsear_region(region))
    teclado = Teclado()
    pila = deque([cap.captura()] * APILAR, maxlen=APILAR)
    periodo = 1.0 / fps
    try:
        while not parar["si"]:
            t0 = time.time()
            pila.append(cap.captura())
            x = torch.as_tensor(np.stack(pila)[None], dtype=torch.float32, device=d)
            with torch.no_grad():
                lt, lx, ly = modelo(x)
                prob = torch.sigmoid(lt)[0].cpu().numpy()
                if usar_raton:
                    # movimiento esperado segun las probabilidades de cada grupo
                    dx = (torch.softmax(lx, 1)[0] * centros).sum().item() * args.raton_escala
                    dy = (torch.softmax(ly, 1)[0] * centros).sum().item() * args.raton_escala
            for k, pr in zip(teclas, prob):
                if k.startswith("clic_") and not usar_raton:
                    continue
                teclado.poner(k, pr > args.umbral)
            if usar_raton:
                teclado.mover(dx if abs(dx) >= 2 else 0, dy if abs(dy) >= 2 else 0)
            time.sleep(max(0.0, periodo - (time.time() - t0)))
    finally:
        teclado.soltar_todo()
        oyente.stop()
        print("Parado.")


if __name__ == "__main__":
    main()
