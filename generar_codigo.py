"""Genera codigo con el mini GPT entrenado por entrenar_codigo.py.

  python generar_codigo.py --prompt "def suma(a, b):"
  python generar_codigo.py --interactivo
"""
import argparse

import torch

from codigo.mini_gpt import MiniGPT
from dispositivo import elegir_dispositivo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="modelos/codigo.pt")
    p.add_argument("--prompt", default="def ")
    p.add_argument("--largo", type=int, default=500, help="bytes a generar")
    p.add_argument("--temperatura", type=float, default=0.8)
    p.add_argument("--interactivo", action="store_true")
    p.add_argument("--dispositivo", default="auto")
    args = p.parse_args()

    d = elegir_dispositivo(args.dispositivo)
    guardado = torch.load(args.modelo, map_location="cpu", weights_only=False)
    modelo = MiniGPT(**guardado["config"])
    modelo.load_state_dict(guardado["modelo"])
    modelo.to(d).eval()

    def completar(prompt):
        idx = torch.tensor([list(prompt.encode("utf-8")) or [10]], device=d)
        salida = modelo.generar(idx, args.largo, args.temperatura)
        return bytes(salida[0].tolist()).decode("utf-8", "replace")

    if not args.interactivo:
        print(completar(args.prompt))
        return
    print("Escribe el inicio del codigo (vacio para salir):")
    while True:
        prompt = input(">>> ")
        if not prompt:
            break
        print(completar(prompt.replace("\\n", "\n")))
        print("-" * 40)


if __name__ == "__main__":
    main()
