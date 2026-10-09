"""Interfaz grafica para usar la IA local sin escribir comandos.

  python interfaz.py
"""
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

RAIZ = os.path.dirname(os.path.abspath(__file__))
DISPOSITIVOS = ["auto", "cuda", "xpu", "mps", "directml", "cpu"]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IA local - juegos y codigo")
        self.geometry("980x800")
        self.minsize(760, 560)
        self.proceso = None
        self.cola = queue.Queue()

        arriba = ttk.Frame(self, padding=(8, 8, 8, 0))
        arriba.pack(fill="x")
        ttk.Label(arriba, text="Dispositivo:").pack(side="left")
        self.dispositivo = tk.StringVar(value="auto")
        ttk.Combobox(arriba, textvariable=self.dispositivo, values=DISPOSITIVOS, width=10,
                     state="readonly").pack(side="left", padx=4)
        ttk.Button(arriba, text="Detectar GPU", command=lambda: self.ejecutar("dispositivo.py", [])).pack(side="left")
        ttk.Label(arriba, text="  auto = usa la GPU (tambien integrada) si la encuentra").pack(side="left")

        panel = ttk.PanedWindow(self, orient="vertical")
        panel.pack(fill="both", expand=True, padx=8, pady=8)
        pestanas = ttk.Notebook(panel)
        panel.add(pestanas, weight=1)

        self.pestana_juegos(pestanas)
        self.pestana_pc(pestanas)
        self.pestana_minigpt(pestanas)
        self.pestana_asistente(pestanas)

        # Consola de salida
        abajo = ttk.Frame(panel)
        panel.add(abajo, weight=1)
        barra = ttk.Frame(abajo)
        barra.pack(fill="x")
        self.estado = tk.StringVar(value="Listo")
        ttk.Label(barra, textvariable=self.estado).pack(side="left")
        ttk.Button(barra, text="Limpiar", command=lambda: self.log.delete("1.0", "end")).pack(side="right")
        ttk.Button(barra, text="Detener", command=self.detener).pack(side="right", padx=4)
        self.log = scrolledtext.ScrolledText(abajo, height=15, font=("Consolas", 10), wrap="word")
        self.log.pack(fill="both", expand=True, pady=4)
        entrada = ttk.Frame(self, padding=(8, 0, 8, 8))
        entrada.pack(fill="x", before=panel, side="bottom")
        ttk.Label(entrada, text="Escribir al programa:").pack(side="left")
        self.texto = tk.StringVar()
        caja = ttk.Entry(entrada, textvariable=self.texto)
        caja.pack(side="left", fill="x", expand=True, padx=4)
        caja.bind("<Return>", lambda e: self.enviar())
        ttk.Button(entrada, text="Enviar", command=self.enviar).pack(side="left")

        self.after(100, self.leer_cola)
        self.protocol("WM_DELETE_WINDOW", self.cerrar)

    # ---------- ayudantes de formulario ----------
    def campo(self, padre, fila, etiqueta, valor="", tipo="texto", opciones=None, ancho=40):
        ttk.Label(padre, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=3)
        var = tk.BooleanVar(value=valor) if tipo == "check" else tk.StringVar(value=str(valor))
        if tipo == "check":
            ttk.Checkbutton(padre, variable=var).grid(row=fila, column=1, sticky="w")
        elif opciones:
            ttk.Combobox(padre, textvariable=var, values=opciones, width=ancho - 2).grid(row=fila, column=1, sticky="w")
        else:
            ttk.Entry(padre, textvariable=var, width=ancho).grid(row=fila, column=1, sticky="w")
        if tipo in ("archivo", "carpeta"):
            def buscar():
                r = filedialog.askdirectory(initialdir=RAIZ) if tipo == "carpeta" else \
                    filedialog.askopenfilename(initialdir=os.path.join(RAIZ, "modelos"))
                if r:
                    var.set(r)
            ttk.Button(padre, text="...", width=3, command=buscar).grid(row=fila, column=2, sticky="w", padx=2)
        return var

    @staticmethod
    def seccion(padre, titulo):
        marco = ttk.LabelFrame(padre, text=titulo, padding=8)
        marco.pack(fill="x", padx=8, pady=6)
        return marco

    # ---------- pestanas ----------
    def pestana_juegos(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Juegos (refuerzo)")
        ttk.Label(f, text="La IA aprende sola probando y recibiendo recompensas. Snake viene incluido; "
                          "tambien cualquier juego de Gymnasium.", wraplength=880).pack(anchor="w", padx=8, pady=4)
        s = self.seccion(f, "Entrenar")
        juego = self.campo(s, 0, "Juego", "snake", opciones=["snake", "CartPole-v1", "LunarLander-v3",
                                                            "Acrobot-v1", "ALE/Pong-v5", "ALE/Breakout-v5"])
        atari = self.campo(s, 1, "Es de Atari", False, tipo="check")
        episodios = self.campo(s, 2, "Episodios", 1500, ancho=10)
        ttk.Button(s, text="Entrenar", command=lambda: self.ejecutar("entrenar_juego.py", [
            "--juego", juego.get(), "--episodios", episodios.get()] + (["--atari"] if atari.get() else []))
        ).grid(row=3, column=1, sticky="w", pady=4)

        s = self.seccion(f, "Ver jugar")
        modelo = self.campo(s, 0, "Modelo", "modelos/snake_mejor.pt", tipo="archivo")
        partidas = self.campo(s, 1, "Partidas", 3, ancho=10)
        ttk.Button(s, text="Jugar", command=lambda: self.ejecutar("jugar.py", [
            "--modelo", modelo.get(), "--partidas", partidas.get()])).grid(row=2, column=1, sticky="w", pady=4)

    def pestana_pc(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Cualquier juego de PC")
        ttk.Label(f, text="La IA te imita: 1) grabas tu pantalla y teclas mientras juegas (F9 empezar/pausar, "
                          "F10 terminar), 2) entrena, 3) juega sola pulsando las teclas (F10 para parar). "
                          "No lo uses en juegos online con anti-trampas: te pueden banear.",
                  wraplength=880).pack(anchor="w", padx=8, pady=4)
        s = self.seccion(f, "1. Grabar")
        nombre = self.campo(s, 0, "Nombre del juego", "mijuego")
        teclas = self.campo(s, 1, "Teclas a aprender", "w,a,s,d,space")
        region = self.campo(s, 2, "Region x,y,ancho,alto", "")
        ttk.Label(s, text="(vacio = pantalla completa)").grid(row=2, column=2, sticky="w")
        fps = self.campo(s, 3, "Fotogramas/seg", 10, ancho=10)
        raton = self.campo(s, 4, "Grabar raton (camara y clics)", False, tipo="check")
        ttk.Button(s, text="Grabar", command=lambda: self.ejecutar("grabar_pc.py", [
            "--nombre", nombre.get(), "--teclas", teclas.get(), "--region", region.get(), "--fps", fps.get()]
            + (["--raton"] if raton.get() else []))
        ).grid(row=5, column=1, sticky="w", pady=4)

        s = self.seccion(f, "2. Entrenar   3. Jugar")
        epocas = self.campo(s, 0, "Epocas", 15, ancho=10)
        ttk.Button(s, text="Entrenar", command=lambda: self.ejecutar("entrenar_pc.py", [
            "--nombre", nombre.get(), "--epocas", epocas.get()])).grid(row=0, column=2, sticky="w", padx=4)
        umbral = self.campo(s, 1, "Umbral para pulsar", 0.5, ancho=10)
        escala = self.campo(s, 2, "Velocidad del raton", 1.0, ancho=10)
        ttk.Button(s, text="Jugar", command=lambda: self.ejecutar("jugar_pc.py", [
            "--modelo", os.path.join("modelos", f"pc_{nombre.get()}.pt"), "--umbral", umbral.get(),
            "--raton-escala", escala.get()])
        ).grid(row=1, column=2, sticky="w", padx=4)

    def pestana_minigpt(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Codigo: mini GPT")
        ttk.Label(f, text="Entrena desde cero un GPT pequeno con tus archivos de codigo. Sirve para aprender "
                          "como funciona; para un asistente util usa la siguiente pestana.",
                  wraplength=880).pack(anchor="w", padx=8, pady=4)
        s = self.seccion(f, "Entrenar")
        datos = self.campo(s, 0, "Carpeta con codigo", "", tipo="carpeta")
        tamano = self.campo(s, 1, "Tamano", "pequeno", opciones=["pequeno", "mediano", "grande"], ancho=12)
        pasos = self.campo(s, 2, "Pasos", 5000, ancho=10)
        ttk.Button(s, text="Entrenar", command=lambda: self.ejecutar("entrenar_codigo.py", [
            "--datos", datos.get(), "--tamano", tamano.get(), "--pasos", pasos.get()], requiere=[datos])
        ).grid(row=3, column=1, sticky="w", pady=4)

        s = self.seccion(f, "Generar")
        prompt = self.campo(s, 0, "Empieza con", "def ", ancho=60)
        largo = self.campo(s, 1, "Largo (letras)", 500, ancho=10)
        ttk.Button(s, text="Generar", command=lambda: self.ejecutar("generar_codigo.py", [
            "--prompt", prompt.get(), "--largo", largo.get()])).grid(row=2, column=1, sticky="w", pady=4)

    def pestana_asistente(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Codigo: asistente")
        ttk.Label(f, text="Usa un modelo de codigo ya entrenado (Qwen2.5-Coder) y ajustalo con tu codigo o con "
                          "ejemplos .jsonl. La primera vez descarga el modelo. Para chatear, escribe abajo en "
                          "'Escribir al programa'.", wraplength=880).pack(anchor="w", padx=8, pady=4)
        modelos = ["Qwen/Qwen2.5-Coder-0.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                   "Qwen/Qwen2.5-Coder-3B-Instruct"]
        s = self.seccion(f, "Ajustar con tus datos (LoRA)")
        base = self.campo(s, 0, "Modelo base", modelos[0], opciones=modelos, ancho=42)
        datos = self.campo(s, 1, "Carpeta de codigo", "", tipo="carpeta")
        ejemplos = self.campo(s, 2, "Ejemplos .jsonl", "ejemplos/ejemplos.jsonl", tipo="archivo")
        pasos = self.campo(s, 3, "Pasos", 300, ancho=10)

        def ajustar():
            rutas = [r for r in (datos.get(), ejemplos.get()) if r.strip()]
            if not rutas:
                messagebox.showwarning("Faltan datos", "Elige una carpeta de codigo o un archivo .jsonl")
                return
            self.ejecutar("finetune_codigo.py", ["--modelo", base.get(), "--pasos", pasos.get(), "--datos", *rutas])
        ttk.Button(s, text="Ajustar", command=ajustar).grid(row=4, column=1, sticky="w", pady=4)

        s = self.seccion(f, "Chatear")
        lora = self.campo(s, 0, "Adaptador LoRA", "modelos/lora_codigo", tipo="carpeta")
        sin_lora = self.campo(s, 1, "Usar modelo base sin ajustar", False, tipo="check")

        def chatear():
            args = ["--modelo", base.get()] if sin_lora.get() else ["--lora", lora.get()]
            self.ejecutar("chat_codigo.py", args)
        ttk.Button(s, text="Iniciar chat", command=chatear).grid(row=2, column=1, sticky="w", pady=4)

    # ---------- procesos ----------
    def ejecutar(self, script, args, requiere=()):
        if any(not v.get().strip() for v in requiere):
            messagebox.showwarning("Faltan datos", "Rellena los campos necesarios")
            return
        if self.proceso and self.proceso.poll() is None:
            if not messagebox.askyesno("Ocupado", "Ya hay algo en marcha. Detenerlo y empezar esto?"):
                return
            self.detener()
        if script not in ("dispositivo.py", "grabar_pc.py"):  # estos no usan la GPU
            args = args + ["--dispositivo", self.dispositivo.get()]
        cmd = [sys.executable, "-u", script] + [str(a) for a in args]
        self.escribir(f"\n$ {' '.join(cmd[2:])}\n")
        entorno = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.proceso = subprocess.Popen(cmd, cwd=RAIZ, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, env=entorno, bufsize=0)
        self.estado.set(f"En marcha: {script}")
        threading.Thread(target=self.lector, args=(self.proceso,), daemon=True).start()

    def lector(self, proceso):
        while True:
            trozo = proceso.stdout.read1(4096) if hasattr(proceso.stdout, "read1") else proceso.stdout.read(1)
            if not trozo:
                break
            self.cola.put(trozo.decode("utf-8", "replace"))
        proceso.wait()
        self.cola.put(("fin", proceso.returncode))

    def leer_cola(self):
        try:
            while True:
                item = self.cola.get_nowait()
                if isinstance(item, tuple):
                    self.estado.set("Listo" if item[1] == 0 else f"Terminado (codigo {item[1]})")
                else:
                    self.escribir(item)
        except queue.Empty:
            pass
        self.after(50, self.leer_cola)

    def escribir(self, texto):
        # \f lo usa jugar.py para "limpiar pantalla" y animar Snake
        if "\f" in texto:
            self.log.delete("1.0", "end")
            self.log.insert("end", texto.rsplit("\f", 1)[1])
            self.log.see("1.0")
            return
        self.log.insert("end", texto)
        self.log.see("end")

    def enviar(self):
        texto = self.texto.get()
        if not (self.proceso and self.proceso.poll() is None):
            return
        self.escribir(texto + "\n")
        try:
            self.proceso.stdin.write((texto + "\n").encode("utf-8"))
            self.proceso.stdin.flush()
        except OSError:
            pass
        self.texto.set("")

    def detener(self):
        if self.proceso and self.proceso.poll() is None:
            # Primero se pide parar con calma (guarda grabaciones y suelta teclas)
            try:
                self.proceso.stdin.write(b"parar\n")
                self.proceso.stdin.flush()
                self.proceso.wait(3)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if self.proceso and self.proceso.poll() is None:
            self.proceso.terminate()
            try:
                self.proceso.wait(5)
            except subprocess.TimeoutExpired:
                self.proceso.kill()
            self.escribir("\n[Detenido]\n")

    def cerrar(self):
        self.detener()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
