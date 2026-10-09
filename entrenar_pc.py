"""Entrena la IA para imitar tus partidas grabadas con grabar_pc.py.

  python entrenar_pc.py --nombre minecraft --epocas 15
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from dispositivo import elegir_dispositivo, describir
from juegos.pc import APILAR, RedImitacion


def cargar(carpeta):
    archivos = sorted(glob.glob(os.path.join(carpeta, "*.npz")))
    if not archivos:
        raise SystemExit(f"No hay grabaciones en {carpeta}. Usa primero grabar_pc.py")
    fotos, etiquetas, indices, teclas, meta = [], [], [], None, {}
    base = 0
    for a in archivos:
        d = np.load(a)
        t = [str(x) for x in d["teclas"]]
        if teclas is None:
            teclas, meta = t, {"fps": float(d["fps"]), "region": str(d["region"])}
        elif t != teclas:
            print(f"Aviso: {a} tiene otras teclas ({t}), se ignora")
            continue
        n = len(d["fotos"])
        fotos.append(d["fotos"])
        etiquetas.append(d["etiquetas"])
        # solo indices con APILAR-1 fotogramas previos dentro de la misma sesion
        indices.extend(range(base + APILAR - 1, base + n))
        base += n
    return np.concatenate(fotos), np.concatenate(etiquetas).astype(np.float32), np.array(indices), teclas, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nombre", required=True)
    p.add_argument("--carpeta", default="datos_pc")
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dispositivo", default="auto")
    p.add_argument("--salida", default=None)
    args = p.parse_args()

    fotos, etiquetas, indices, teclas, meta = cargar(os.path.join(args.carpeta, args.nombre))
    print(f"{len(indices)} ejemplos, teclas {teclas}")
    d = elegir_dispositivo(args.dispositivo)
    print(f"Dispositivo: {describir(d)}")

    rng = np.random.default_rng(0)
    rng.shuffle(indices)
    n_val = max(1, len(indices) // 10)
    val, ent = indices[:n_val], indices[n_val:]

    # Las teclas poco pulsadas pesan mas, si no la IA aprende a no pulsarlas nunca
    frec = etiquetas[ent].mean(0).clip(0.01, 0.99)
    peso_pos = torch.tensor((1 - frec) / frec, dtype=torch.float32, device=d).clamp(max=10)

    modelo = RedImitacion(len(teclas)).to(d)
    optim = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)

    def lote(idx):
        x = np.stack([fotos[i - APILAR + 1:i + 1] for i in idx])
        if modelo.training:  # pequena variacion de brillo para generalizar mejor
            x = np.clip(x.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255)
        return (torch.as_tensor(x, dtype=torch.float32, device=d),
                torch.as_tensor(etiquetas[idx], device=d))

    salida = args.salida or os.path.join("modelos", f"pc_{args.nombre}.pt")
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    mejor = float("inf")
    inicio = time.time()
    for ep in range(1, args.epocas + 1):
        modelo.train()
        rng.shuffle(ent)
        total = 0.0
        for i in range(0, len(ent), args.lote):
            x, y = lote(ent[i:i + args.lote])
            perdida = F.binary_cross_entropy_with_logits(modelo(x), y, pos_weight=peso_pos)
            optim.zero_grad()
            perdida.backward()
            optim.step()
            total += perdida.item() * len(x)
        modelo.eval()
        aciertos, pv = [], 0.0
        with torch.no_grad():
            for i in range(0, len(val), args.lote):
                x, y = lote(val[i:i + args.lote])
                out = modelo(x)
                pv += F.binary_cross_entropy_with_logits(out, y, pos_weight=peso_pos).item() * len(x)
                aciertos.append(((out > 0).float() == y).float().cpu().numpy())
        pv /= len(val)
        acc = np.concatenate(aciertos).mean(0)
        print(f"Epoca {ep:3d} | perdida {total / len(ent):.4f} | validacion {pv:.4f} | acierto por tecla "
              + " ".join(f"{k}:{a * 100:.0f}%" for k, a in zip(teclas, acc)) + f" | {time.time() - inicio:.0f}s",
              flush=True)
        if pv < mejor:
            mejor = pv
            torch.save({"modelo": modelo.state_dict(), "teclas": teclas, **meta}, salida)

    print(f"\nListo. Modelo guardado en {salida}")
    print(f"Para que juegue: python jugar_pc.py --modelo {salida}")


if __name__ == "__main__":
    main()
