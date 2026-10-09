"""La IA se entrena SOLA jugando a cualquier juego de PC (aprendizaje por refuerzo).

Nadie le dice que hacer: prueba acciones y aprende de una recompensa.
La recompensa se calcula mirando la pantalla:
  * Curiosidad: premio por ver cosas nuevas (explorar) y castigo por quedarse
    atascada (la pantalla no cambia). Funciona en cualquier juego sin configurar.
  * Barra (opcional): una zona de la pantalla con un color, p. ej. la barra de
    vida o de experiencia. Si hay mas de ese color -> premio; si hay menos -> castigo.
  * Objetos (opcional, muy util): cosas que le ensenas marcandolas en pantalla
    (madera, piedra...) con ensenar_objeto.py o la interfaz. La IA ve donde estan
    y gana premio al buscarlas, acercarse, apuntarles y picarlas/recogerlas.

  python autoentrenar_pc.py --nombre minecraft --minutos 60
  python autoentrenar_pc.py --nombre minecraft --desde-imitacion
  python autoentrenar_pc.py --nombre juego --barra 20,40,200,10 --barra-color 0,200,0
  python ensenar_objeto.py --nombre minecraft --objeto madera --zona 600,300,40,40   (y luego entrenar)

Teclas: por defecto usa las que tu usaste en tus grabaciones de ese juego; si no hay
grabaciones, un conjunto amplio de teclas tipicas de juego. Tambien se puede dar una lista.

  F9 = pausar/continuar (para arreglar algo a mano)   F10 = parar y guardar
Despues, para verla jugar sin seguir aprendiendo:  ... --solo-jugar
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dispositivo import elegir_dispositivo, describir
from juegos.objetos import Detector, DetectorAuto, objetos_de
from juegos.ventana import region_texto, ventana_activa
from juegos.pc import (APILAR, CLICS, PELIGROSAS, TAM, TECLAS_JUEGO, Capturador, RedImitacion, Teclado,
                       avisar_al_parar, nombre_tecla, parsear_region, parsear_teclas, teclas_grabadas)

GIROS = (8, 25, 60)  # pixeles por accion: apuntar fino, girar, girar rapido


def elegir_teclas(texto, nombre, carpeta_datos):
    teclas = parsear_teclas(texto)
    if teclas is not None:
        return [k for k in teclas if k not in PELIGROSAS], "las que has indicado"
    ruta_imit = os.path.join("modelos", f"pc_{nombre}.pt")
    if os.path.exists(ruta_imit):
        t = torch.load(ruta_imit, map_location="cpu", weights_only=False)["teclas"]
        t = [k for k in t if not k.startswith("clic_") and k not in PELIGROSAS]
        if t:
            return t, f"las de tu modelo de imitacion ({ruta_imit})"
    t = teclas_grabadas(os.path.join(carpeta_datos, nombre))
    if t:
        return t, "las que usaste en tus grabaciones"
    return list(TECLAS_JUEGO), "teclas tipicas de juego (no hay grabaciones)"


def crear_acciones(teclas, raton):
    """Lista de acciones posibles. Cada una = (teclas que pulsa, movimiento dx, dy)."""
    acciones = [((), 0, 0)]  # no hacer nada
    acciones += [((k,), 0, 0) for k in teclas]
    if "w" in teclas:  # combinaciones utiles al andar
        acciones += [(("w", k), 0, 0) for k in teclas if k in ("space", "shift", "ctrl", "a", "d")]
    if raton:
        for g in GIROS:  # mirar a los lados en varias velocidades
            acciones += [((), -g, 0), ((), g, 0)]
        for g in GIROS[:2]:  # arriba/abajo, menos rapido
            acciones += [((), 0, -g // 2), ((), 0, g // 2)]
        acciones += [((c,), 0, 0) for c in CLICS.values()]
        if "w" in teclas:  # andar girando
            acciones += [(("w",), -GIROS[1], 0), (("w",), GIROS[1], 0)]
        acciones += [(("clic_izq",), -GIROS[0], 0), (("clic_izq",), GIROS[0], 0)]  # picar apuntando
    return acciones


def crear_acciones_v1(teclas, raton):
    """Acciones de la primera version (modelos guardados antes de que se guardara la lista)."""
    acciones = [((), 0, 0)] + [((k,), 0, 0) for k in teclas]
    if "w" in teclas:
        acciones += [(("w", k), 0, 0) for k in teclas if k in ("space", "shift", "ctrl", "a", "d")]
    if raton:
        acciones += [((), -30, 0), ((), 30, 0), ((), 0, -15), ((), 0, 15)]
        acciones += [((c,), 0, 0) for c in CLICS.values()]
        if "w" in teclas:
            acciones += [(("w",), -30, 0), (("w",), 30, 0)]
    return acciones


def acciones_guardadas(guardado):
    """Las acciones con las que se entreno un modelo guardado (para poder continuarlo)."""
    if "acciones" in guardado:
        return guardado["acciones"]
    n = guardado["red"]["cabeza.2.weight"].shape[0]
    for crear in (crear_acciones_v1, crear_acciones):
        acciones = crear(guardado["teclas"], guardado["raton"])
        if len(acciones) == n:
            return acciones
    return None


class RedQ(nn.Module):
    """Misma 'vista' (capas convolucionales) que la red de imitacion, para poder reutilizarla.
    Con objetos ensenados recibe un canal mas: el mapa de donde estan los objetos."""

    def __init__(self, n_acciones, canales=APILAR):
        super().__init__()
        base = RedImitacion(1)
        self.conv = base.conv
        if canales != APILAR:
            self.conv[0] = nn.Conv2d(canales, 32, 8, stride=4)
        n = base.cabeza[0].in_features
        self.cabeza = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), nn.Linear(512, n_acciones))

    def forward(self, x):
        return self.cabeza(self.conv(x / 255.0))


def cargar_red(red, estado, con_cabeza=True):
    """Carga pesos aunque cambie el numero de canales de entrada (p. ej. al anadir objetos).
    Los canales que coinciden se copian; los nuevos empiezan en cero."""
    propio = red.state_dict()
    for k, v in estado.items():
        if k not in propio or (not con_cabeza and not k.startswith("conv.")):
            continue
        if propio[k].shape == v.shape:
            propio[k] = v
        elif k == "conv.0.weight":
            c = min(v.shape[1], propio[k].shape[1])
            nuevo = torch.zeros_like(propio[k])
            nuevo[:, :c] = v[:, :c]
            propio[k] = nuevo
    red.load_state_dict(propio)


class Curiosidad(nn.Module):
    """Random Network Distillation: una red fija al azar y otra que intenta imitarla.
    Si falla mucho, es que la imagen es nueva -> premio."""

    def __init__(self):
        super().__init__()

        def red():
            return nn.Sequential(nn.Conv2d(1, 16, 8, 4), nn.LeakyReLU(), nn.Conv2d(16, 32, 4, 2), nn.LeakyReLU(),
                                 nn.Flatten(), nn.LazyLinear(128))
        self.fija = red()
        self.alumno = red()
        with torch.no_grad():
            self.fija(torch.zeros(1, 1, TAM, TAM))
            self.alumno(torch.zeros(1, 1, TAM, TAM))
        for p in self.fija.parameters():
            p.requires_grad_(False)

    def forward(self, foto):  # foto: (B, 1, TAM, TAM)
        x = foto / 255.0
        return ((self.alumno(x) - self.fija(x)) ** 2).mean(1)


class Memoria:
    """Guarda cada fotograma una sola vez (ahorra mucha RAM) y arma las pilas al sacar muestras."""

    def __init__(self, capacidad, con_mascara=False):
        self.cap = capacidad
        self.fotos = np.zeros((capacidad, TAM, TAM), np.uint8)
        self.mascaras = np.zeros((capacidad, TAM, TAM), np.uint8) if con_mascara else None
        self.acc = np.zeros(capacidad, np.int64)
        self.rec = np.zeros(capacidad, np.float32)
        self.corte = np.ones(capacidad, bool)  # True = no se puede apilar hacia atras desde aqui
        self.i = 0
        self.n = 0

    def nueva_foto(self, foto, corte=False, mascara=None):
        self.fotos[self.i] = foto
        if self.mascaras is not None:
            self.mascaras[self.i] = 255 * mascara if mascara is not None else 0
        self.corte[self.i] = corte
        idx = self.i
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)
        return idx

    def apuntar(self, idx, accion, recompensa):
        self.acc[idx] = accion
        self.rec[idx] = recompensa

    def _pila(self, idx):
        ids = [idx]
        for _ in range(APILAR - 1):
            if self.corte[ids[0]]:
                ids.insert(0, ids[0])
            else:
                ids.insert(0, (ids[0] - 1) % self.cap)
        if self.mascaras is None:
            return self.fotos[ids]
        # canal extra: donde estan los objetos ensenados en el fotograma actual
        return np.concatenate([self.fotos[ids], self.mascaras[idx][None]])

    def muestra(self, lote):
        validos = []
        while len(validos) < lote:
            j = random.randrange(self.n)
            sig = (j + 1) % self.cap
            if sig == self.i or self.corte[sig]:  # el siguiente aun no existe o es de otra racha
                continue
            validos.append(j)
        obs = np.stack([self._pila(j) for j in validos])
        sig = np.stack([self._pila((j + 1) % self.cap) for j in validos])
        return obs, self.acc[validos], self.rec[validos], sig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", required=True)
    p.add_argument("--teclas", default="auto", help="'auto' (por defecto) o una lista: w,a,s,d,space")
    p.add_argument("--sin-raton", action="store_true", help="no mover la camara ni hacer clic")
    p.add_argument("--raton", action="store_true", help=argparse.SUPPRESS)  # antiguo: ahora es lo normal
    p.add_argument("--carpeta", default="datos_pc", help=argparse.SUPPRESS)
    p.add_argument("--region", default="", help="x,y,ancho,alto del juego, 'auto' = la ventana donde hagas clic "
                                                "(vacio = pantalla completa)")
    p.add_argument("--fps", type=float, default=5, help="decisiones por segundo")
    p.add_argument("--minutos", type=float, default=30)
    p.add_argument("--barra", default="", help="x,y,ancho,alto de una barra (vida, experiencia, puntos...)")
    p.add_argument("--barra-color", default="0,200,0", help="color r,g,b de la barra")
    p.add_argument("--barra-invertida", action="store_true", help="MENOS color es mejor (p. ej. barra de dano)")
    p.add_argument("--peso-curiosidad", type=float, default=1.0)
    p.add_argument("--objetos", default="todos",
                   help="objetos ensenados a buscar: 'todos' (por defecto), una lista 'madera,piedra' o 'ninguno'")
    p.add_argument("--peso-objetos", type=float, default=1.0)
    p.add_argument("--sin-auto", action="store_true",
                   help="no buscar 'cosas interesantes' automaticamente (colores raros en el juego)")
    p.add_argument("--peso-auto", type=float, default=0.5)
    p.add_argument("--desde-imitacion", action="store_true",
                   help="empezar con la 'vista' aprendida en modelos/pc_<nombre>.pt")
    p.add_argument("--solo-jugar", action="store_true", help="no aprender, solo jugar con lo aprendido")
    p.add_argument("--azar-pasos", type=int, default=20000,
                   help="pasos en los que va dejando de actuar al azar (explorar -> aprovechar)")
    p.add_argument("--memoria", type=int, default=30000, help="fotogramas guardados (30000 ~ 280 MB)")
    p.add_argument("--espera", type=int, default=5)
    p.add_argument("--dispositivo", default="auto")
    args = p.parse_args()

    from pynput import keyboard

    d = elegir_dispositivo(args.dispositivo)
    ruta = os.path.join("modelos", f"auto_{args.nombre}.pt")
    os.makedirs("modelos", exist_ok=True)

    if os.path.exists(ruta):
        guardado = torch.load(ruta, map_location="cpu", weights_only=False)
        teclas, raton = guardado["teclas"], guardado["raton"]
        region = args.region or guardado["region"]  # la zona indicada ahora manda (la ventana pudo moverse)
        print(f"Continuando el entrenamiento de {ruta} ({guardado['pasos']} pasos previos)")
    else:
        if args.solo_jugar:
            raise SystemExit(f"No existe {ruta}. Entrena primero sin --solo-jugar.")
        guardado = None
        teclas, origen = elegir_teclas(args.teclas, args.nombre, args.carpeta)
        print(f"Teclas: {', '.join(teclas)}  <- {origen}")
        raton, region = not args.sin_raton, args.region
    # al continuar se usan las mismas acciones con las que se entreno
    acciones = acciones_guardadas(guardado) if guardado else None
    solo_vista = False
    if acciones is None:
        acciones = crear_acciones(teclas, raton)
        solo_vista = guardado is not None

    print(f"Dispositivo: {describir(d)} | {len(acciones)} acciones posibles")

    # Objetos ensenados (madera, piedra...)
    objetos = {} if args.objetos.strip().lower() in ("ninguno", "no", "") else objetos_de(args.nombre)
    if args.objetos.strip().lower() not in ("todos", "ninguno", "no", ""):
        pedidos = [o.strip() for o in args.objetos.split(",")]
        faltan = [o for o in pedidos if o not in objetos]
        if faltan:
            print(f"Aviso: no has ensenado {', '.join(faltan)} (usa ensenar_objeto.py)")
        objetos = {o: h for o, h in objetos.items() if o in pedidos}
    detector = Detector(objetos) if objetos else None
    auto = None if args.sin_auto else DetectorAuto()
    nombres_obj = (detector.nombres if detector else []) + (["cosas nuevas"] if auto else [])
    pesos_obj = [args.peso_objetos] * (len(nombres_obj) - (1 if auto else 0)) + ([args.peso_auto] if auto else [])
    if nombres_obj:
        print("Busca: " + ", ".join(nombres_obj)
              + ("  (cosas nuevas = colores raros en este juego, se detectan solas)" if auto else ""))
    canales = APILAR + (1 if nombres_obj else 0)

    red = RedQ(len(acciones), canales).to(d)
    objetivo = RedQ(len(acciones), canales).to(d)
    curiosidad = Curiosidad().to(d)
    pasos = 0
    if guardado and not solo_vista:
        cargar_red(red, guardado["red"])
        curiosidad.load_state_dict(guardado["curiosidad"])
        pasos = guardado["pasos"]
    elif guardado:
        # no se sabe con que acciones se entreno: se conserva lo que aprendio a "ver"
        # y la parte que decide la accion empieza de nuevo
        cargar_red(red, guardado["red"], con_cabeza=False)
        curiosidad.load_state_dict(guardado["curiosidad"])
        print("Aviso: el modelo guardado usaba otras acciones; se conserva su vista y se reajusta el resto")
    elif args.desde_imitacion:
        ruta_imit = os.path.join("modelos", f"pc_{args.nombre}.pt")
        if os.path.exists(ruta_imit):
            estado = torch.load(ruta_imit, map_location="cpu", weights_only=False)["modelo"]
            cargar_red(red, estado, con_cabeza=False)
            print(f"Empieza con la vista aprendida en {ruta_imit}")
        else:
            print(f"Aviso: no existe {ruta_imit}, empieza de cero")
    objetivo.load_state_dict(red.state_dict())
    optim = torch.optim.Adam(red.parameters(), lr=1e-4)
    optim_cur = torch.optim.Adam(curiosidad.alumno.parameters(), lr=1e-4)

    barra = parsear_region(args.barra)
    if args.barra_invertida and not barra:
        print("Aviso: --barra-invertida no hace nada sin --barra (marca la zona de la barra)")
    color = [int(c) for c in args.barra_color.split(",")]

    estado = {"fin": False, "pausa": False}

    def al_pulsar(t):
        n = nombre_tecla(t)
        if n == "f10":
            estado["fin"] = True
            return False
        if n == "f9":
            estado["pausa"] = not estado["pausa"]
            print("EN PAUSA (F9 para seguir)" if estado["pausa"] else "SIGUE", flush=True)

    oyente = keyboard.Listener(on_press=al_pulsar)
    oyente.start()
    avisar_al_parar(estado)

    for s in range(args.espera, 0, -1):
        if estado["fin"]:
            return
        print(f"Empieza en {s}... (haz clic en la ventana del juego)", flush=True)
        time.sleep(1)
    print("EN MARCHA. F9 pausa, F10 para y guarda.", flush=True)

    if region.strip().lower() == "auto":
        titulo, zona = ventana_activa()
        if not zona or zona[2] < 50 or zona[3] < 50:
            print("Aviso: no se pudo detectar la ventana del juego; se usa la pantalla completa")
            region = ""
        else:
            region = region_texto(zona)
            print(f"Juego detectado: '{titulo}' en la zona {region}")
    cap = Capturador(parsear_region(region))
    teclado = Teclado()
    memoria = Memoria(args.memoria, con_mascara=bool(nombres_obj))
    periodo = 1.0 / args.fps
    fin_tiempo = time.time() + args.minutos * 60
    media_cur, var_cur = 0.0, 1.0
    quieto = 0
    nivel_barra = cap.fraccion_color(barra, color) if barra else 0.0
    ult_guardado = ult_aviso = time.time()
    suma_rec, n_rec = 0.0, 0
    n_obj = len(nombres_obj)
    medidas = np.zeros((n_obj, 3))      # visible, centro, mira de cada objeto
    recogidos = np.zeros(n_obj, int)
    golpes = np.zeros(n_obj, int)       # fotogramas seguidos golpeando cada objeto

    def mirar():
        """Captura la pantalla; con objetos, tambien donde estan y cuanto se ven."""
        if not n_obj:
            return cap.captura(), None, medidas
        gris, rgb = cap.captura_con_color()
        mascaras = list(detector.detectar(rgb)) if detector else []
        if auto:
            mascaras.append(auto.detectar(rgb))
        mascaras = np.array(mascaras)
        m = np.array([Detector.medir(mk) for mk in mascaras])
        return gris, mascaras.any(0), m

    def guardar():
        if not args.solo_jugar:
            torch.save({"red": red.state_dict(), "curiosidad": curiosidad.state_dict(), "teclas": teclas,
                        "raton": raton, "region": region, "pasos": pasos, "acciones": acciones,
                        "canales": canales, "objetos": nombres_obj}, ruta)

    foto, mascara, medidas = mirar()
    idx = memoria.nueva_foto(foto, corte=True, mascara=mascara)
    try:
        while not estado["fin"] and time.time() < fin_tiempo:
            t0 = time.time()
            if estado["pausa"]:
                teclado.soltar_todo()
                time.sleep(0.2)
                foto, mascara, medidas = mirar()
                idx = memoria.nueva_foto(foto, corte=True, mascara=mascara)  # tras la pausa, racha nueva
                continue

            # 1. Elegir accion (al principio mucho al azar, luego cada vez menos)
            eps = 0.02 if args.solo_jugar else max(0.05, 1.0 - pasos / args.azar_pasos)
            if random.random() < eps:
                a = random.randrange(len(acciones))
            else:
                with torch.no_grad():
                    x = torch.as_tensor(memoria._pila(idx)[None], dtype=torch.float32, device=d)
                    a = int(red(x).argmax(1).item())
            pulsa, dx, dy = acciones[a]

            # 2. Hacerla en el juego
            for k in teclas + ([c for c in CLICS.values()] if raton else []):
                teclado.poner(k, k in pulsa)
            teclado.mover_suave(dx, dy, max(0.0, periodo - (time.time() - t0)))

            # 3. Mirar el resultado y calcular la recompensa
            nueva, mascara, nuevas = mirar()
            with torch.no_grad():
                cur = curiosidad(torch.as_tensor(nueva[None, None], dtype=torch.float32, device=d)).item()
            media_cur = 0.99 * media_cur + 0.01 * cur
            var_cur = 0.99 * var_cur + 0.01 * (cur - media_cur) ** 2
            r = args.peso_curiosidad * float(np.clip((cur - media_cur) / (var_cur ** 0.5 + 1e-6), -1, 3)) * 0.1
            cambio = np.abs(nueva.astype(np.int16) - foto.astype(np.int16)).mean()
            quieto = quieto + 1 if cambio < 1.0 else 0
            picando = n_obj > 0 and nuevas[:, 2].max() > 0.2  # quieta pero apuntando a un objeto: bien
            if quieto > 3 * args.fps and not picando:  # atascada mas de 3 segundos
                r -= 0.2
            for i in range(n_obj):
                (vis0, cen0, mir0), (vis, cen, mir) = medidas[i], nuevas[i]
                # Solo cuenta MEJORAR (verlo mas grande, mas centrado): quedarse mirando no da nada.
                # (Si diera premio por tenerlo en la mira, la IA aprende a mirarlo para siempre sin picar)
                ro = float(np.clip((vis - vis0) * 10, -0.3, 0.5))
                ro += float(np.clip((cen - cen0) + (mir - mir0), -0.5, 0.5))
                # golpeando sin mover la camara ni andar: si desaparece de la mira es que lo ha
                # picado/recogido (si no, la IA haria trampa girando la camara mientras hace clic)
                solo_clic = set(pulsa) == {"clic_izq"} and dx == 0 and dy == 0
                if solo_clic and mir0 > 0.3:
                    golpes[i] += 1
                    if golpes[i] <= 6 * args.fps:                   # premio por golpear, solo unos segundos
                        ro += 0.05
                    if mir < mir0 * 0.5:                            # ha desaparecido de la mira: recogido
                        ro += 3.0
                        recogidos[i] += 1
                        golpes[i] = 0
                elif mir0 <= 0.3:
                    golpes[i] = 0
                r += pesos_obj[i] * ro
            medidas = nuevas
            if barra:
                nivel = cap.fraccion_color(barra, color)
                delta = (nivel - nivel_barra) * (-1 if args.barra_invertida else 1)
                # Un salto enorme de golpe es un reinicio (subir de nivel, reaparecer...): no cuenta
                if abs(delta) < 0.3:
                    r += float(np.clip(delta * 20, -2, 2))
                nivel_barra = nivel
            memoria.apuntar(idx, a, r)
            foto = nueva
            idx = memoria.nueva_foto(foto, mascara=mascara)
            suma_rec += r
            n_rec += 1
            pasos += 1

            # 4. Aprender (DQN doble + curiosidad)
            if not args.solo_jugar and memoria.n > 500:
                obs, acc, rec, sig = memoria.muestra(32)
                obs = torch.as_tensor(obs, dtype=torch.float32, device=d)
                sig = torch.as_tensor(sig, dtype=torch.float32, device=d)
                acc = torch.as_tensor(acc, device=d)
                rec = torch.as_tensor(rec, device=d)
                q = red(obs).gather(1, acc[:, None]).squeeze(1)
                with torch.no_grad():
                    mejor = red(sig).argmax(1, keepdim=True)
                    meta = rec + 0.97 * objetivo(sig).gather(1, mejor).squeeze(1)
                perdida = F.smooth_l1_loss(q, meta)
                optim.zero_grad()
                perdida.backward()
                nn.utils.clip_grad_norm_(red.parameters(), 10)
                optim.step()
                perdida_cur = curiosidad(sig[:, APILAR - 1:APILAR]).mean()
                optim_cur.zero_grad()
                perdida_cur.backward()
                optim_cur.step()
                if pasos % 1000 == 0:
                    objetivo.load_state_dict(red.state_dict())

            if time.time() - ult_aviso > 15:
                txt_barra = f" | barra {nivel_barra * 100:.0f}%" if barra else ""
                txt_barra += "".join(f" | {n}: ve {medidas[i, 0] * 100:.0f}% mira {medidas[i, 2] * 100:.0f}%"
                                     f" recogidos {recogidos[i]}" for i, n in enumerate(nombres_obj))
                print(f"paso {pasos:7d} | recompensa media {suma_rec / max(n_rec, 1):+.3f} | azar {eps:.2f}"
                      f"{txt_barra} | quedan {max(0, fin_tiempo - time.time()) / 60:.0f} min", flush=True)
                suma_rec, n_rec, ult_aviso = 0.0, 0, time.time()
            if time.time() - ult_guardado > 120:
                guardar()
                ult_guardado = time.time()
    finally:
        teclado.soltar_todo()
        oyente.stop()
        guardar()
        if not args.solo_jugar:
            print(f"Guardado en {ruta} ({pasos} pasos). Si lo vuelves a lanzar, sigue desde aqui.")


if __name__ == "__main__":
    main()
