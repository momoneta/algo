"""Entrena una IA para jugar con aprendizaje por refuerzo (DQN).

Ejemplos:
  python entrenar_juego.py --juego snake --episodios 2000
  python entrenar_juego.py --juego CartPole-v1 --episodios 500
  python entrenar_juego.py --juego LunarLander-v3 --episodios 1500
  python entrenar_juego.py --juego ALE/Breakout-v5 --atari --pasos 2000000

Se puede usar cualquier juego de Gymnasium con acciones discretas.
"""
import argparse
import os
import time
from collections import deque

import numpy as np

from dispositivo import elegir_dispositivo, describir
from juegos.dqn import AgenteDQN


def crear_entorno(nombre, atari=False, render_mode=None):
    if nombre.lower() == "snake":
        from juegos.snake_env import SnakeEnv
        return SnakeEnv(render_mode=render_mode)

    import gymnasium as gym
    if atari:
        import ale_py  # noqa: F401  (registra los juegos ALE/...)
        gym.register_envs(ale_py)
        env = gym.make(nombre, frameskip=1, render_mode=render_mode)
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env
    return gym.make(nombre, render_mode=render_mode)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--juego", default="snake", help="snake o un id de Gymnasium (CartPole-v1, ...)")
    p.add_argument("--atari", action="store_true", help="aplica el preprocesado de Atari (imagenes)")
    p.add_argument("--episodios", type=int, default=1000)
    p.add_argument("--pasos", type=int, default=0, help="limite total de pasos (0 = sin limite)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--eps-inicio", type=float, default=1.0)
    p.add_argument("--eps-fin", type=float, default=0.02)
    p.add_argument("--eps-pasos", type=int, default=50_000, help="pasos para bajar epsilon")
    p.add_argument("--calentamiento", type=int, default=1000, help="pasos aleatorios antes de aprender")
    p.add_argument("--dispositivo", default="auto", help="auto|cuda|xpu|mps|directml|cpu")
    p.add_argument("--salida", default="modelos")
    p.add_argument("--continuar", help="ruta de un modelo .pt para seguir entrenando")
    args = p.parse_args()

    if args.atari:
        args.lr = min(args.lr, 1e-4)
        args.lote = max(args.lote, 32)

    dispositivo = elegir_dispositivo(args.dispositivo)
    print(f"Dispositivo: {describir(dispositivo)}")

    env = crear_entorno(args.juego, args.atari)
    forma = env.observation_space.shape
    n_acciones = env.action_space.n
    print(f"Juego: {args.juego} | observacion {forma} | {n_acciones} acciones")

    if args.continuar:
        agente, _ = AgenteDQN.cargar(args.continuar, dispositivo)
        agente.memoria.datos = deque(maxlen=args.buffer)
        print(f"Continuando desde {args.continuar}")
    else:
        agente = AgenteDQN(forma, n_acciones, lr=args.lr, gamma=args.gamma, buffer=args.buffer,
                           lote=args.lote, dispositivo=dispositivo)

    os.makedirs(args.salida, exist_ok=True)
    nombre = args.juego.replace("/", "_")
    ruta_mejor = os.path.join(args.salida, f"{nombre}_mejor.pt")
    ruta_final = os.path.join(args.salida, f"{nombre}_final.pt")
    extra = {"juego": args.juego, "atari": args.atari}

    pasos = 0
    recompensas = deque(maxlen=100)
    mejor_media = -float("inf")
    inicio = time.time()

    for ep in range(1, args.episodios + 1):
        obs, _ = env.reset()
        total, fin = 0.0, False
        while not fin:
            eps = max(args.eps_fin, args.eps_inicio - (args.eps_inicio - args.eps_fin) * pasos / args.eps_pasos)
            if args.continuar:
                eps = args.eps_fin
            accion = agente.actuar(obs, eps)
            sig, r, terminado, truncado, _ = env.step(accion)
            # Para Atari se guardan uint8 para ahorrar memoria
            o = np.asarray(obs, dtype=np.uint8 if args.atari else np.float32)
            s = np.asarray(sig, dtype=np.uint8 if args.atari else np.float32)
            agente.memoria.guardar(o, accion, float(np.sign(r)) if args.atari else r, s, terminado)
            obs = sig
            total += r
            pasos += 1
            fin = terminado or truncado
            if pasos > args.calentamiento:
                agente.aprender()

        recompensas.append(total)
        media = float(np.mean(recompensas))
        if len(recompensas) >= 10 and media > mejor_media:
            mejor_media = media
            agente.guardar(ruta_mejor, extra)
        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep:5d} | pasos {pasos:8d} | recompensa {total:8.1f} | media100 {media:8.2f} "
                  f"| eps {eps:.3f} | {time.time() - inicio:6.0f}s")
        if args.pasos and pasos >= args.pasos:
            break

    agente.guardar(ruta_final, extra)
    env.close()
    print(f"\nListo. Mejor media: {mejor_media:.2f}")
    print(f"Modelos guardados en: {ruta_mejor} y {ruta_final}")
    print(f"Para verlo jugar: python jugar.py --modelo {ruta_mejor}")


if __name__ == "__main__":
    main()
