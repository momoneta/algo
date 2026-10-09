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
from juegos.pc import APILAR, CENTROS_MOV_ANTIGUOS, PELIGROSAS, Capturador, decidir_mov, RedImitacion, Teclado, avisar_al_parar, nombre_tecla, parsear_region


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True)
    p.add_argument("--umbral", type=float, default=0.5, help="probabilidad minima para pulsar una tecla")
    p.add_argument("--region", default=None, help="por defecto la misma que al grabar")
    p.add_argument("--espera", type=int, default=5, help="segundos antes de empezar")
    p.add_argument("--raton-escala", type=float, default=1.0,
                   help="multiplica el movimiento del raton (sube si gira poco, baja si gira demasiado)")
    p.add_argument("--sin-raton", action="store_true", help="no mover el raton aunque el modelo sepa")
    p.add_argument("--suavizado", type=float, default=0.5,
                   help="0 = sin suavizar, 0.9 = muy suave (menos temblores, reacciona mas lento)")
    p.add_argument("--dispositivo", default="auto")
    args = p.parse_args()

    from pynput import keyboard

    datos = torch.load(args.modelo, map_location="cpu", weights_only=False)
    teclas = datos["teclas"]
    d = elegir_dispositivo(args.dispositivo)
    raton = datos.get("raton", False)
    # los modelos antiguos usaban menos grupos de movimiento
    centros_np = datos.get("centros_mov", CENTROS_MOV_ANTIGUOS.tolist())
    modelo = RedImitacion(len(teclas), raton, len(centros_np))
    modelo.load_state_dict(datos["modelo"])
    modelo.to(d).eval()
    region = args.region if args.region is not None else datos.get("region", "")
    fps = datos.get("fps", 10)
    usar_raton = raton and not args.sin_raton
    print(f"Dispositivo: {describir(d)} | teclas {teclas} | {fps} fps"
          + (" | mueve el raton" if usar_raton else ""))
    centros = torch.as_tensor(centros_np, dtype=torch.float32, device=d)
    suave_x = suave_y = 0.0

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
                    dx = decidir_mov(lx, centros).item() * args.raton_escala
                    dy = decidir_mov(ly, centros).item() * args.raton_escala
            for k, pr in zip(teclas, prob):
                if k in PELIGROSAS or (k.startswith("clic_") and not usar_raton):
                    continue
                teclado.poner(k, pr > args.umbral)
            resto = max(0.0, periodo - (time.time() - t0))
            if usar_raton:
                # suavizado: mezcla con el movimiento anterior para quitar temblores
                suave_x = args.suavizado * suave_x + (1 - args.suavizado) * dx
                suave_y = args.suavizado * suave_y + (1 - args.suavizado) * dy
                mx = suave_x if abs(suave_x) >= 1 else 0
                my = suave_y if abs(suave_y) >= 1 else 0
                # el giro se reparte durante todo el fotograma: camara fluida, no a saltos
                teclado.mover_suave(mx, my, resto)
            else:
                time.sleep(resto)
    finally:
        teclado.soltar_todo()
        oyente.stop()
        print("Parado.")


if __name__ == "__main__":
    main()
