"""Entrena desde cero un mini GPT con tus propios archivos (codigo o texto).

  python entrenar_codigo.py --datos C:/mis_proyectos --tamano pequeno --pasos 5000
  python entrenar_codigo.py --datos ./mi_codigo --extensiones .py,.js --continuar modelos/codigo.pt

Despues genera con:  python generar_codigo.py --prompt "def ordenar("
"""
import argparse
import os
import time

import torch

from codigo.mini_gpt import MiniGPT, TAMANOS
from dispositivo import elegir_dispositivo, describir

EXT_POR_DEFECTO = (".py,.js,.ts,.jsx,.tsx,.java,.c,.cpp,.h,.hpp,.cs,.go,.rs,.rb,.php,.lua,"
                   ".html,.css,.sql,.sh,.kt,.swift,.gd,.md,.txt")
CARPETAS_IGNORADAS = {".git", "node_modules", "__pycache__", "venv", ".venv", "build", "dist", "modelos"}


def leer_datos(rutas, extensiones, max_mb):
    trozos, total = [], 0
    for ruta in rutas:
        archivos = [ruta] if os.path.isfile(ruta) else []
        for raiz, dirs, nombres in os.walk(ruta):
            dirs[:] = [d for d in dirs if d not in CARPETAS_IGNORADAS]
            archivos += [os.path.join(raiz, n) for n in nombres if n.lower().endswith(extensiones)]
        for a in archivos:
            try:
                with open(a, "rb") as f:
                    contenido = f.read()
            except OSError:
                continue
            if b"\x00" in contenido[:1024]:  # binario
                continue
            trozos.append(contenido + b"\n\n")
            total += len(contenido)
            if total > max_mb * 1024 * 1024:
                return b"".join(trozos), len(trozos)
    return b"".join(trozos), len(trozos)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", nargs="+", required=True, help="carpetas o archivos para aprender")
    p.add_argument("--extensiones", default=EXT_POR_DEFECTO)
    p.add_argument("--tamano", choices=TAMANOS.keys(), default="pequeno")
    p.add_argument("--pasos", type=int, default=5000)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-mb", type=float, default=200, help="maximo de datos a cargar")
    p.add_argument("--dispositivo", default="auto", help="auto|cuda|xpu|mps|directml|cpu")
    p.add_argument("--salida", default="modelos/codigo.pt")
    p.add_argument("--continuar", help="modelo .pt para seguir entrenando")
    args = p.parse_args()

    exts = tuple(e.strip().lower() for e in args.extensiones.split(",") if e.strip())
    texto, n_archivos = leer_datos(args.datos, exts, args.max_mb)
    if len(texto) < 10_000:
        raise SystemExit(f"Solo hay {len(texto)} bytes de datos. Necesitas mas archivos (minimo ~10 KB, ideal varios MB).")
    print(f"Datos: {n_archivos} archivos, {len(texto) / 1e6:.2f} MB")

    dispositivo = elegir_dispositivo(args.dispositivo)
    print(f"Dispositivo: {describir(dispositivo)}")

    datos = torch.frombuffer(bytearray(texto), dtype=torch.uint8).long()
    corte = int(len(datos) * 0.95)
    entreno, valid = datos[:corte], datos[corte:]

    if args.continuar:
        guardado = torch.load(args.continuar, map_location="cpu", weights_only=False)
        modelo = MiniGPT(**guardado["config"])
        modelo.load_state_dict(guardado["modelo"])
        print(f"Continuando desde {args.continuar}")
    else:
        capas, cabezas, dim, ctx = TAMANOS[args.tamano]
        modelo = MiniGPT(capas, cabezas, dim, ctx)
    modelo.to(dispositivo)
    ctx = modelo.contexto
    print(f"Modelo: {modelo.n_parametros() / 1e6:.1f}M parametros, contexto {ctx}")

    optim = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.1)
    usar_amp = getattr(dispositivo, "type", "") in ("cuda", "xpu")

    def lote(fuente):
        i = torch.randint(len(fuente) - ctx - 1, (args.lote,))
        x = torch.stack([fuente[j:j + ctx] for j in i])
        y = torch.stack([fuente[j + 1:j + ctx + 1] for j in i])
        return x.to(dispositivo), y.to(dispositivo)

    @torch.no_grad()
    def evaluar():
        modelo.eval()
        perdidas = [modelo(*lote(valid if len(valid) > ctx + 1 else entreno))[1].item() for _ in range(20)]
        modelo.train()
        return sum(perdidas) / len(perdidas)

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    mejor = float("inf")
    inicio = time.time()
    for paso in range(1, args.pasos + 1):
        # calentamiento + decaimiento coseno del learning rate
        lr = args.lr * min(1.0, paso / 200) * (0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(paso / args.pasos * 3.14159)).item()))
        for g in optim.param_groups:
            g["lr"] = lr
        x, y = lote(entreno)
        if usar_amp:
            with torch.autocast(device_type=dispositivo.type, dtype=torch.bfloat16):
                _, perdida = modelo(x, y)
        else:
            _, perdida = modelo(x, y)
        optim.zero_grad(set_to_none=True)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        optim.step()

        if paso % 100 == 0 or paso == args.pasos:
            pv = evaluar()
            print(f"paso {paso:6d} | perdida {perdida.item():.3f} | validacion {pv:.3f} | {time.time() - inicio:6.0f}s")
            if pv < mejor:
                mejor = pv
                torch.save({"modelo": modelo.state_dict(), "config": modelo.config}, args.salida)
        if paso % 1000 == 0:
            muestra = modelo.generar(torch.tensor([list(b"def ")], device=dispositivo), 120)
            print("--- muestra ---\n" + bytes(muestra[0].tolist()).decode("utf-8", "replace") + "\n---------------")

    print(f"\nListo. Modelo guardado en {args.salida} (mejor validacion {mejor:.3f})")
    print(f'Prueba: python generar_codigo.py --modelo {args.salida} --prompt "def "')


if __name__ == "__main__":
    main()
