# IA local: aprende a jugar y a programar

Proyecto para entrenar tu propia IA **en tu PC**, sin servicios en la nube.
Funciona con CPU, GPU dedicada y **GPU integrada** (Intel Iris Xe / Arc, AMD Radeon, Apple M).

**¿No te gustan los comandos?** Abre la interfaz gráfica y hazlo todo con botones
(incluye elegir la zona del juego y la barra arrastrando con el ratón):

```bash
python interfaz.py
```

| Qué quieres | Script | Cómo aprende |
|---|---|---|
| Que juegue a un juego | `entrenar_juego.py` → `jugar.py` | Aprendizaje por refuerzo (DQN) |
| Que juegue a **cualquier juego de tu PC** | `grabar_pc.py` → `entrenar_pc.py` → `jugar_pc.py` | Te imita: aprende de tus partidas grabadas |
| Un modelo de código desde cero | `entrenar_codigo.py` → `generar_codigo.py` | Mini GPT entrenado con tus archivos |
| Un asistente de código bueno | `finetune_codigo.py` → `chat_codigo.py` | Ajuste LoRA de un modelo ya preentrenado (Qwen2.5-Coder) |

---

## 1. Instalación

Necesitas Python 3.10 o superior.

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

Instala PyTorch **según tu GPU**:

| Tu GPU | Comando |
|---|---|
| **Intel integrada** (Iris Xe, UHD, Arc) — Windows o Linux | `pip install torch --index-url https://download.pytorch.org/whl/xpu` |
| **AMD integrada** (Radeon Vega/680M/780M) o cualquier GPU — **Windows** | `pip install torch-directml` |
| AMD en **Linux** (ROCm) | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| NVIDIA | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Apple M1/M2/M3 | `pip install torch` |
| Solo CPU | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

Para Intel instala también los drivers de gráficos actualizados de Intel.

Luego el resto:

```bash
pip install -r requirements.txt
```

Comprueba qué dispositivo detecta:

```bash
python dispositivo.py
```

Todos los scripts eligen el dispositivo automáticamente (`cuda` → `xpu` → `mps` → `directml` → `cpu`).
Puedes forzarlo con `--dispositivo xpu` (Intel), `--dispositivo directml` (AMD/Intel en Windows) o `--dispositivo cpu`.

---

## 2. IA que juega

Usa **DQN** (Deep Q-Network): la IA prueba acciones, recibe recompensas y aprende qué hacer.

```bash
# Snake incluido (no necesita nada extra)
python entrenar_juego.py --juego snake --episodios 2000
python jugar.py --modelo modelos/snake_mejor.pt

# Cualquier juego de Gymnasium con acciones discretas
python entrenar_juego.py --juego CartPole-v1 --episodios 500
python entrenar_juego.py --juego LunarLander-v3 --episodios 1500   # requiere: pip install "gymnasium[box2d]"
python jugar.py --modelo modelos/LunarLander-v3_mejor.pt

# Atari (pip install ale-py) — necesita muchas horas, mejor con GPU
python entrenar_juego.py --juego ALE/Pong-v5 --atari --episodios 100000 --pasos 2000000 --buffer 200000 --eps-pasos 500000
```

Seguir entrenando un modelo: `--continuar modelos/snake_mejor.pt`.

**Tu propio juego:** crea una clase con `reset()` y `step(accion)` como en
`juegos/snake_env.py` (devuelve observación, recompensa, terminado, truncado, info)
y añádela en `crear_entorno()` de `entrenar_juego.py`. La clave es diseñar bien la
**recompensa**: premia lo que quieres que haga y castiga lo que no.

---

## 2b. IA que juega a cualquier juego de PC (por imitación)

Funciona con casi cualquier juego (Minecraft, juegos de carreras, plataformas, emuladores…):

1. **Grabar**: juegas tú y se guardan la pantalla, **todas las teclas** que pulses, la cámara y los clics.
   ```bash
   python grabar_pc.py --nombre minecraft --region 0,0,1280,720 --fps 15
   ```
   `F9` empieza/pausa, `F10` termina y guarda. Puedes grabar varias sesiones; se juntan todas.
   No hace falta decir qué teclas: se aprenden todas las que uses (se ignoran las casi no usadas y
   Esc, F4, F9, F10 y la tecla de Windows). Si prefieres limitarlo: `--teclas w,a,s,d,space`.
   Sin ratón: `--sin-raton`.
2. **Entrenar**: la red aprende "con esta imagen, se pulsan estas teclas".
   ```bash
   python entrenar_pc.py --nombre minecraft --epocas 15
   ```
3. **Jugar**: la IA mira la pantalla y pulsa las teclas sola. Tienes 5 s para hacer clic en el juego; `F10` la para.
   ```bash
   python jugar_pc.py --modelo modelos/pc_minecraft.pt
   ```

Consejos:
- Pon el juego **en ventana** y usa `--region x,y,ancho,alto` con la zona del juego (más rápido y preciso).
- Graba **mucho** (30–60 min) y juega siempre de forma parecida: la IA copia lo que ve, también tus errores.
- Puedes grabar varias sesiones aunque uses teclas distintas: al entrenar se juntan todas.
- Si la IA no pulsa casi nada, baja `--umbral` (p. ej. 0.3); si pulsa demasiado, súbelo.
- En Windows, `pydirectinput` hace que funcione con juegos DirectX que ignoran las teclas simuladas normales.
- Ratón: la cámara se mueve de forma fluida (el giro se reparte durante cada fotograma) y suavizada.
  Si gira demasiado poco o demasiado, ajusta `--raton-escala` (p. ej. 2 o 0.5); si tiembla, sube
  `--suavizado` (0 a 0.9).
  Con `--sin-raton` juega solo con el teclado. No cambies la sensibilidad del ratón del juego entre grabar y jugar.
- Aprende lo que ve en una imagen pequeña (96×96 en grises), así que funciona mejor en juegos de reacción
  (esquivar, conducir, saltar, minar) que en juegos de estrategia o con mucho texto.
- **No lo uses en juegos online con anti-trampas**: te pueden banear la cuenta.

### Que aprenda SOLA (sin grabarte)

`autoentrenar_pc.py` deja a la IA jugando por su cuenta (aprendizaje por refuerzo). Aprende de una recompensa
que saca de la pantalla:

- **Curiosidad** (siempre): premio por ver cosas nuevas, castigo si se queda atascada (la pantalla no cambia).
- **Barra** (opcional, recomendado): una zona de la pantalla con un color, como la barra de experiencia o de vida.
  Si hay más de ese color, premio. Usa `--barra-invertida` si menos color es mejor.

```bash
# Minecraft: explorar con curiosidad, empezando desde lo que aprendió imitándote
# (usa solas las teclas de tus grabaciones; si no hay, un conjunto amplio de teclas típicas de juego)
python autoentrenar_pc.py --nombre Minecraft --region 0,0,1280,720 --desde-imitacion --minutos 60

# Con la barra de experiencia verde de Minecraft como objetivo (ajusta la zona a tu pantalla)
python autoentrenar_pc.py --nombre Minecraft --barra 450,650,380,6 --barra-color 128,255,32

# Verla jugar con lo aprendido, sin seguir aprendiendo
python autoentrenar_pc.py --nombre Minecraft --solo-jugar
```

- `F9` pausa (por si muere o se atasca y quieres arreglarlo a mano), `F10` para y guarda.
- Si lo lanzas otra vez, **continúa** donde lo dejó (`modelos/auto_<nombre>.pt`).
- Al principio actúa casi al azar y va dejando de hacerlo durante los primeros ~20 000 pasos (~1 h a 5 por segundo).
  Necesita **horas**: déjalo jugando en un mundo que no te importe.
- Para encontrar la zona de la barra: haz una captura de pantalla y mira las coordenadas en Paint.

---

## 3. IA de código

### Opción A — Mini GPT desde cero (aprende de tus archivos)

```bash
python entrenar_codigo.py --datos C:/ruta/a/tus/proyectos --tamano pequeno --pasos 5000
python generar_codigo.py --prompt "def ordenar_lista("
python generar_codigo.py --interactivo
```

- `--tamano pequeno` (~3M parámetros) va bien en GPU integrada; `mediano` y `grande` necesitan más.
- Cuantos **más datos** mejor (varios MB de código). Aprende el estilo y la sintaxis, pero un modelo
  tan pequeño no "razona": es ideal para aprender cómo funciona un GPT por dentro.

### Opción B — Ajustar un modelo de código real (recomendado para uso práctico)

Parte de **Qwen2.5-Coder-0.5B-Instruct** (ya sabe programar) y lo ajusta con LoRA a tus datos.
La primera vez descarga el modelo (~1 GB).

```bash
# Probar el modelo base sin entrenar
python chat_codigo.py

# Entrenar con tu código y/o con ejemplos pregunta-respuesta
python finetune_codigo.py --datos ./mi_codigo ejemplos.jsonl --pasos 300
python chat_codigo.py --lora modelos/lora_codigo
```

Formato de `ejemplos.jsonl` (una línea por ejemplo; hay uno en `ejemplos/ejemplos.jsonl`):

```json
{"pregunta": "Como leo un archivo en Python?", "respuesta": "with open('a.txt') as f:\n    texto = f.read()"}
```

Memoria aproximada para el fine-tuning:

| Modelo | RAM/VRAM | GPU integrada |
|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct | ~4 GB | Sí |
| Qwen2.5-Coder-1.5B-Instruct | ~8 GB | Intel Arc / con 16 GB de RAM |
| Qwen2.5-Coder-3B / 7B | 12–24 GB | Mejor GPU dedicada |

Las GPUs integradas usan la RAM del sistema: si tienes 16 GB o más, puedes subir de tamaño.

---

## Consejos para GPU integrada

- **Intel**: usa el PyTorch `xpu`; es el soporte oficial y el más rápido para Iris Xe/Arc.
- **AMD en Windows**: `torch-directml` funciona bien para el juego y el mini GPT. Para
  `finetune_codigo.py` algunas operaciones pueden no estar soportadas; si falla, usa `--dispositivo cpu`.
- Si te quedas sin memoria: baja `--lote`, `--contexto` o usa un modelo más pequeño.
- Juegos pequeños (Snake, CartPole) entrenan a menudo **igual o más rápido en CPU**, porque la red es
  diminuta. La GPU se nota en Atari y en los modelos de código.

## Estructura

```
interfaz.py           interfaz gráfica con todo lo de abajo
grabar_pc.py          graba tus partidas de cualquier juego de PC
entrenar_pc.py        entrena la IA que te imita
jugar_pc.py           la IA juega sola pulsando teclas
autoentrenar_pc.py    la IA aprende sola jugando (refuerzo con curiosidad / barra)
juegos/pc.py          captura de pantalla, red de imitación y teclado
dispositivo.py        detección de CPU / GPU (incluidas integradas)
entrenar_juego.py     entrena la IA de juegos (DQN)
jugar.py              mira jugar a la IA
juegos/dqn.py         agente DQN (redes MLP y CNN)
juegos/snake_env.py   juego Snake incluido
entrenar_codigo.py    entrena el mini GPT con tus archivos
generar_codigo.py     genera código con el mini GPT
codigo/mini_gpt.py    modelo transformer
finetune_codigo.py    ajuste LoRA de un modelo de código preentrenado
chat_codigo.py        chat con el modelo de código
```
