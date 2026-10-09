"""Mira jugar a una IA ya entrenada.

  python jugar.py --modelo modelos/snake_mejor.pt
  python jugar.py --modelo modelos/CartPole-v1_mejor.pt --partidas 3
"""
import argparse
import os
import time

from dispositivo import elegir_dispositivo
from entrenar_juego import crear_entorno
from juegos.dqn import AgenteDQN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", required=True)
    p.add_argument("--partidas", type=int, default=3)
    p.add_argument("--velocidad", type=float, default=0.08, help="segundos entre pasos (snake)")
    p.add_argument("--sin-ventana", action="store_true", help="no mostrar el juego, solo puntuaciones")
    p.add_argument("--dispositivo", default="auto")
    args = p.parse_args()

    agente, extra = AgenteDQN.cargar(args.modelo, elegir_dispositivo(args.dispositivo))
    juego = extra.get("juego", "snake")
    es_snake = juego.lower() == "snake"
    modo = None if args.sin_ventana or es_snake else "human"
    env = crear_entorno(juego, extra.get("atari", False), render_mode=modo)

    for partida in range(1, args.partidas + 1):
        obs, _ = env.reset()
        total, fin = 0.0, False
        while not fin:
            obs, r, terminado, truncado, info = env.step(agente.actuar(obs))
            total += r
            fin = terminado or truncado
            if es_snake and not args.sin_ventana:
                os.system("cls" if os.name == "nt" else "clear")
                print(env.render())
                time.sleep(args.velocidad)
        extra_txt = f" | puntos {info['puntos']}" if "puntos" in info else ""
        print(f"Partida {partida}: recompensa {total:.1f}{extra_txt}")
    env.close()


if __name__ == "__main__":
    main()
