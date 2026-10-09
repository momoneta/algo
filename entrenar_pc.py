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
from juegos.pc import APILAR, CENTROS_MOV, PELIGROSAS, RedImitacion, decidir_mov, objetivo_suave, ordenar_teclas


def cargar(carpeta, min_fotogramas):
    """Junta todas las grabaciones aunque usen teclas distintas (se unen todas)."""
    archivos = sorted(glob.glob(os.path.join(carpeta, "*.npz")))
    if not archivos:
        raise SystemExit(f"No hay grabaciones en {carpeta}. Usa primero grabar_pc.py")
    sesiones = [dict(np.load(a)) for a in archivos]
    teclas = ordenar_teclas(str(k) for d in sesiones for k in d["teclas"])

    fotos, etiquetas, movs, con_raton, indices = [], [], [], [], []
    base = 0
    for d in sesiones:
        n = len(d["fotos"])
        tabla = np.zeros((n, len(teclas)), np.float32)
        for j, k in enumerate(str(x) for x in d["teclas"]):
            tabla[:, teclas.index(k)] = d["etiquetas"][:, j]
        raton = bool(d.get("raton", False))
        fotos.append(d["fotos"])
        etiquetas.append(tabla)
        movs.append(d["mov"] if "mov" in d else np.zeros((n, 2), np.float32))
        con_raton.append(np.full(n, raton))
        # solo indices con APILAR-1 fotogramas previos dentro de la misma sesion
        indices.extend(range(base + APILAR - 1, base + n))
        base += n
    etiquetas = np.concatenate(etiquetas)

    # Teclas casi sin usar o peligrosas: fuera (no se pueden aprender y ensucian)
    usos = etiquetas.sum(0)
    quedan = [i for i, k in enumerate(teclas) if usos[i] >= min_fotogramas and k not in PELIGROSAS]
    fuera = [f"{teclas[i]}({int(usos[i])})" for i in range(len(teclas)) if i not in quedan]
    if fuera:
        print("Teclas descartadas (poco usadas o peligrosas): " + ", ".join(fuera))
    teclas = [teclas[i] for i in quedan]
    etiquetas = etiquetas[:, quedan]

    mov = np.concatenate(movs)
    # cada movimiento se asigna al grupo (clase) mas cercano de CENTROS_MOV
    clases = np.abs(mov[:, :, None] - CENTROS_MOV[None, None, :]).argmin(2)
    con_raton = np.concatenate(con_raton)
    ultimo = sesiones[-1]
    meta = {"fps": float(ultimo["fps"]), "region": str(ultimo["region"]), "raton": bool(con_raton.any()),
            "centros_mov": CENTROS_MOV.tolist()}
    return np.concatenate(fotos), etiquetas, clases, con_raton, np.array(indices), teclas, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nombre", required=True)
    p.add_argument("--carpeta", default="datos_pc")
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dispositivo", default="auto")
    p.add_argument("--min-fotogramas", type=int, default=5,
                   help="teclas pulsadas en menos fotogramas que esto se ignoran")
    p.add_argument("--salida", default=None)
    args = p.parse_args()

    fotos, etiquetas, clases, con_raton, indices, teclas, meta = cargar(
        os.path.join(args.carpeta, args.nombre), args.min_fotogramas)
    raton = meta["raton"]
    print(f"{len(indices)} ejemplos ({len(indices) / meta['fps'] / 60:.1f} min) | {len(teclas)} teclas: "
          + ", ".join(teclas) + (" + movimiento del raton" if raton else ""))
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
        cuenta = np.bincount(clases[ent][con_raton[ent]].ravel(), minlength=len(CENTROS_MOV)) + 1
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
                torch.as_tensor(clases[idx], dtype=torch.long, device=d),
                torch.as_tensor(con_raton[idx], dtype=torch.float32, device=d))

    def perdida_mov(logits, c, m):
        # etiqueta suave + peso por grupo; solo cuentan los fotogramas grabados con raton
        t = objetivo_suave(c, logits.shape[1])
        por_ejemplo = -(t * F.log_softmax(logits, 1)).sum(1) * peso_mov[c]
        return (por_ejemplo * m).sum() / m.sum().clamp(min=1)

    def calcular_perdida(salida, y, c, m):
        lt, lx, ly = salida
        perdida = F.binary_cross_entropy_with_logits(lt, y, pos_weight=peso_pos)
        if raton:
            perdida = perdida + 0.5 * (perdida_mov(lx, c[:, 0], m) + perdida_mov(ly, c[:, 1], m))
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
            x, y, c, m = lote(ent[i:i + args.lote])
            perdida = calcular_perdida(modelo(x), y, c, m)
            optim.zero_grad()
            perdida.backward()
            optim.step()
            total += perdida.item() * len(x)
        modelo.eval()
        aciertos, error_raton, pv = [], [], 0.0
        centros = torch.as_tensor(CENTROS_MOV, device=d)
        with torch.no_grad():
            for i in range(0, len(val), args.lote):
                x, y, c, m = lote(val[i:i + args.lote])
                out = modelo(x)
                pv += calcular_perdida(out, y, c, m).item() * len(x)
                aciertos.append(((out[0] > 0).float() == y).float().cpu().numpy())
                if raton:
                    for eje, logits in ((0, out[1]), (1, out[2])):
                        pred = decidir_mov(logits, centros)
                        err = (pred - centros[c[:, eje]]).abs()
                        if m.sum() > 0:
                            error_raton.append(((err * m).sum() / m.sum()).item())
        pv /= len(val)
        acc = np.concatenate(aciertos).mean(0)
        texto_raton = f" | error raton {np.mean(error_raton):.1f}px" if raton and error_raton else ""
        if len(teclas) <= 8:
            texto_teclas = "acierto " + " ".join(f"{k}:{a * 100:.0f}%" for k, a in zip(teclas, acc))
        else:  # con muchas teclas, solo la media y las 3 peores
            peores = sorted(zip(acc, teclas))[:3]
            texto_teclas = (f"acierto medio {acc.mean() * 100:.0f}% (peores: "
                            + " ".join(f"{k}:{a * 100:.0f}%" for a, k in peores) + ")")
        print(f"Epoca {ep:3d} | perdida {total / len(ent):.4f} | validacion {pv:.4f} | " + texto_teclas + texto_raton
              + f" | {time.time() - inicio:.0f}s",
              flush=True)
        if pv < mejor:
            mejor = pv
            torch.save({"modelo": modelo.state_dict(), "teclas": teclas, **meta}, salida)

    print(f"\nListo. Modelo guardado en {salida}")
    print(f"Para que juegue: python jugar_pc.py --modelo {salida}")


if __name__ == "__main__":
    main()
