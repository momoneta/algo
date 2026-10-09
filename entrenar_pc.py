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
from juegos.pc import APILAR, CENTROS_MOV, RedImitacion


def cargar(carpeta):
    archivos = sorted(glob.glob(os.path.join(carpeta, "*.npz")))
    if not archivos:
        raise SystemExit(f"No hay grabaciones en {carpeta}. Usa primero grabar_pc.py")
    fotos, etiquetas, movs, indices, teclas, meta = [], [], [], [], None, {}
    base = 0
    for a in archivos:
        d = np.load(a)
        t = [str(x) for x in d["teclas"]]
        if teclas is None:
            raton = bool(d["raton"]) if "raton" in d else False
            teclas, meta = t, {"fps": float(d["fps"]), "region": str(d["region"]), "raton": raton}
        elif t != teclas:
            print(f"Aviso: {a} tiene otras teclas ({t}), se ignora")
            continue
        n = len(d["fotos"])
        fotos.append(d["fotos"])
        etiquetas.append(d["etiquetas"])
        movs.append(d["mov"] if "mov" in d else np.zeros((n, 2), np.float32))
        # solo indices con APILAR-1 fotogramas previos dentro de la misma sesion
        indices.extend(range(base + APILAR - 1, base + n))
        base += n
    mov = np.concatenate(movs)
    # cada movimiento se asigna al grupo (clase) mas cercano de CENTROS_MOV
    clases = np.abs(mov[:, :, None] - CENTROS_MOV[None, None, :]).argmin(2)
    return (np.concatenate(fotos), np.concatenate(etiquetas).astype(np.float32), clases,
            np.array(indices), teclas, meta)


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

    fotos, etiquetas, clases, indices, teclas, meta = cargar(os.path.join(args.carpeta, args.nombre))
    raton = meta["raton"]
    print(f"{len(indices)} ejemplos, teclas {teclas}" + (" + movimiento del raton" if raton else ""))
    d = elegir_dispositivo(args.dispositivo)
    print(f"Dispositivo: {describir(d)}")

    rng = np.random.default_rng(0)
    rng.shuffle(indices)
    n_val = max(1, len(indices) // 10)
    val, ent = indices[:n_val], indices[n_val:]

    # Las teclas poco pulsadas pesan mas, si no la IA aprende a no pulsarlas nunca
    frec = etiquetas[ent].mean(0).clip(0.01, 0.99)
    peso_pos = torch.tensor((1 - frec) / frec, dtype=torch.float32, device=d).clamp(max=10)

    # Igual con el raton: "no moverse" es lo mas comun, asi que los movimientos pesan mas
    peso_mov = None
    if raton:
        cuenta = np.bincount(clases[ent].ravel(), minlength=len(CENTROS_MOV)) + 1
        w = (cuenta / cuenta.sum()) ** -0.5
        peso_mov = torch.tensor(w / w.mean(), dtype=torch.float32, device=d)

    modelo = RedImitacion(len(teclas), raton).to(d)
    optim = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)

    def lote(idx):
        x = np.stack([fotos[i - APILAR + 1:i + 1] for i in idx])
        if modelo.training:  # pequena variacion de brillo para generalizar mejor
            x = np.clip(x.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255)
        return (torch.as_tensor(x, dtype=torch.float32, device=d),
                torch.as_tensor(etiquetas[idx], device=d),
                torch.as_tensor(clases[idx], dtype=torch.long, device=d))

    def calcular_perdida(salida, y, c):
        lt, lx, ly = salida
        perdida = F.binary_cross_entropy_with_logits(lt, y, pos_weight=peso_pos)
        if raton:
            perdida = perdida + 0.5 * (F.cross_entropy(lx, c[:, 0], weight=peso_mov)
                                       + F.cross_entropy(ly, c[:, 1], weight=peso_mov))
        return perdida

    salida = args.salida or os.path.join("modelos", f"pc_{args.nombre}.pt")
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    mejor = float("inf")
    inicio = time.time()
    for ep in range(1, args.epocas + 1):
        modelo.train()
        rng.shuffle(ent)
        total = 0.0
        for i in range(0, len(ent), args.lote):
            x, y, c = lote(ent[i:i + args.lote])
            perdida = calcular_perdida(modelo(x), y, c)
            optim.zero_grad()
            perdida.backward()
            optim.step()
            total += perdida.item() * len(x)
        modelo.eval()
        aciertos, error_raton, pv = [], [], 0.0
        centros = torch.as_tensor(CENTROS_MOV, device=d)
        with torch.no_grad():
            for i in range(0, len(val), args.lote):
                x, y, c = lote(val[i:i + args.lote])
                out = modelo(x)
                pv += calcular_perdida(out, y, c).item() * len(x)
                aciertos.append(((out[0] > 0).float() == y).float().cpu().numpy())
                if raton:
                    for eje, logits in ((0, out[1]), (1, out[2])):
                        pred = (F.softmax(logits, 1) * centros).sum(1)
                        error_raton.append((pred - centros[c[:, eje]]).abs().mean().item())
        pv /= len(val)
        acc = np.concatenate(aciertos).mean(0)
        texto_raton = f" | error raton {np.mean(error_raton):.1f}px" if raton else ""
        print(f"Epoca {ep:3d} | perdida {total / len(ent):.4f} | validacion {pv:.4f} | acierto por tecla "
              + " ".join(f"{k}:{a * 100:.0f}%" for k, a in zip(teclas, acc)) + texto_raton
              + f" | {time.time() - inicio:.0f}s",
              flush=True)
        if pv < mejor:
            mejor = pv
            torch.save({"modelo": modelo.state_dict(), "teclas": teclas, **meta}, salida)

    print(f"\nListo. Modelo guardado en {salida}")
    print(f"Para que juegue: python jugar_pc.py --modelo {salida}")


if __name__ == "__main__":
    main()
