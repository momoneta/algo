"""Interfaz grafica para usar la IA local sin escribir comandos.

  python interfaz.py
"""
import glob
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)  # para importar juegos/ aunque se lance desde otra carpeta
DISPOSITIVOS = ["auto", "cuda", "xpu", "mps", "directml", "cpu"]

# En Windows con pantallas escaladas (125%, 150%...) sin esto las coordenadas no coinciden
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

# Colores (tema oscuro)
C = {
    "fondo": "#16181d", "panel": "#1f2229", "tarjeta": "#262a33", "borde": "#323743",
    "texto": "#e6e8ee", "suave": "#9aa1b1", "acento": "#5b8cff", "acento2": "#4a78e6",
    "ok": "#3ecf8e", "aviso": "#f5a524", "error": "#f2555a", "campo": "#1b1e24", "consola": "#0f1115",
}
FUENTE = ("Segoe UI", 10) if sys.platform == "win32" else ("DejaVu Sans", 10)
MONO = ("Consolas", 10) if sys.platform == "win32" else ("DejaVu Sans Mono", 10)


def estilos(raiz):
    s = ttk.Style(raiz)
    s.theme_use("clam")
    raiz.configure(bg=C["fondo"])
    s.configure(".", background=C["fondo"], foreground=C["texto"], font=FUENTE, bordercolor=C["borde"],
                fieldbackground=C["campo"], troughcolor=C["panel"], focuscolor=C["acento"])
    s.configure("TFrame", background=C["fondo"])
    s.configure("Panel.TFrame", background=C["panel"])
    s.configure("Tarjeta.TFrame", background=C["tarjeta"])
    s.configure("TLabel", background=C["fondo"], foreground=C["texto"])
    s.configure("Panel.TLabel", background=C["panel"])
    s.configure("Tarjeta.TLabel", background=C["tarjeta"])
    s.configure("Suave.TLabel", background=C["tarjeta"], foreground=C["suave"])
    s.configure("SuaveFondo.TLabel", background=C["fondo"], foreground=C["suave"])
    s.configure("Titulo.TLabel", background=C["fondo"], font=(FUENTE[0], 16, "bold"))
    s.configure("Logo.TLabel", background=C["panel"], font=(FUENTE[0], 14, "bold"), foreground=C["acento"])
    s.configure("Paso.TLabel", background=C["tarjeta"], font=(FUENTE[0], 11, "bold"))
    s.configure("Num.TLabel", background=C["acento"], foreground="white", font=(FUENTE[0], 10, "bold"),
                padding=(7, 1))
    s.configure("TButton", background=C["borde"], foreground=C["texto"], borderwidth=0, padding=(12, 6))
    s.map("TButton", background=[("active", "#3c4250"), ("disabled", C["panel"])],
          foreground=[("disabled", C["suave"])])
    s.configure("Acento.TButton", background=C["acento"], foreground="white", font=(FUENTE[0], 10, "bold"))
    s.map("Acento.TButton", background=[("active", C["acento2"])])
    s.configure("Peligro.TButton", background="#4a2a2e", foreground="#ffb3b6")
    s.map("Peligro.TButton", background=[("active", "#5c3035")])
    s.configure("Menu.TButton", background=C["panel"], foreground=C["suave"], anchor="w", padding=(16, 10),
                font=(FUENTE[0], 11))
    s.map("Menu.TButton", background=[("active", C["tarjeta"])], foreground=[("active", C["texto"])])
    s.configure("MenuActivo.TButton", background=C["tarjeta"], foreground=C["texto"], anchor="w",
                padding=(16, 10), font=(FUENTE[0], 11, "bold"))
    s.configure("TEntry", fieldbackground=C["campo"], foreground=C["texto"], insertcolor=C["texto"],
                bordercolor=C["borde"], lightcolor=C["borde"], darkcolor=C["borde"], padding=5)
    s.configure("TCombobox", fieldbackground=C["campo"], foreground=C["texto"], background=C["borde"],
                arrowcolor=C["texto"], bordercolor=C["borde"], padding=4)
    s.map("TCombobox", fieldbackground=[("readonly", C["campo"]), ("!disabled", C["campo"])],
          foreground=[("readonly", C["texto"]), ("!disabled", C["texto"])],
          selectbackground=[("readonly", C["campo"])], selectforeground=[("readonly", C["texto"])])
    for nombre in ("TScrollbar", "Vertical.TScrollbar"):
        s.configure(nombre, background=C["borde"], troughcolor=C["consola"], arrowcolor=C["suave"],
                    bordercolor=C["fondo"], lightcolor=C["borde"], darkcolor=C["borde"], gripcount=0)
        s.map(nombre, background=[("active", "#4a5160"), ("!active", C["borde"])])
    raiz.option_add("*TCombobox*Listbox.background", C["campo"])
    raiz.option_add("*TCombobox*Listbox.foreground", C["texto"])
    raiz.option_add("*TCombobox*Listbox.selectBackground", C["acento"])
    s.configure("TCheckbutton", background=C["tarjeta"], foreground=C["texto"], indicatorbackground=C["campo"],
                indicatorforeground=C["acento"])
    s.map("TCheckbutton", background=[("active", C["tarjeta"])],
          indicatorbackground=[("selected", C["acento"])])
    s.configure("Horizontal.TProgressbar", background=C["acento"], troughcolor=C["panel"], bordercolor=C["panel"],
                lightcolor=C["acento"], darkcolor=C["acento"])


# ---------- selector de zona de pantalla ----------
def elegir_zona(app, titulo, al_terminar):
    """Pantalla semitransparente: arrastra con el raton para marcar una zona. Esc cancela."""
    app.withdraw()
    app.after(300, lambda: _abrir_selector(app, titulo, al_terminar))


def _abrir_selector(app, titulo, al_terminar):
    sel = tk.Toplevel(app)
    sel.overrideredirect(True)
    ancho, alto = sel.winfo_screenwidth(), sel.winfo_screenheight()
    sel.geometry(f"{ancho}x{alto}+0+0")
    sel.attributes("-topmost", True)
    try:
        sel.attributes("-alpha", 0.35)
    except tk.TclError:
        pass
    lienzo = tk.Canvas(sel, bg="black", highlightthickness=0, cursor="crosshair")
    lienzo.pack(fill="both", expand=True)
    lienzo.create_text(ancho // 2, 40, text=titulo + "   (arrastra con el raton · Esc = cancelar)",
                       fill="white", font=(FUENTE[0], 16, "bold"))
    datos = {"ini": None, "rect": None}

    def terminar(zona):
        sel.destroy()

        def despues():
            # primero se usa la zona (p. ej. capturar el objeto) y despues vuelve la ventana,
            # para no fotografiar la propia ventana de la IA
            try:
                if zona:
                    al_terminar(zona)
            finally:
                app.deiconify()
        app.after(250, despues)

    def pulsar(e):
        datos["ini"] = (e.x_root, e.y_root)
        datos["rect"] = lienzo.create_rectangle(e.x, e.y, e.x, e.y, outline="#5b8cff", width=3)

    def arrastrar(e):
        if datos["rect"]:
            x0, y0 = datos["ini"]
            lienzo.coords(datos["rect"], x0 - sel.winfo_rootx(), y0 - sel.winfo_rooty(), e.x, e.y)

    def soltar(e):
        if not datos["ini"]:
            return
        x0, y0 = datos["ini"]
        x, y = min(x0, e.x_root), min(y0, e.y_root)
        w, h = abs(e.x_root - x0), abs(e.y_root - y0)
        terminar((x, y, w, h) if w > 3 and h > 3 else None)

    lienzo.bind("<ButtonPress-1>", pulsar)
    lienzo.bind("<B1-Motion>", arrastrar)
    lienzo.bind("<ButtonRelease-1>", soltar)
    sel.bind("<Escape>", lambda e: terminar(None))
    sel.focus_force()


def color_de_barra(zona):
    """El color 'de la barra' en una zona: el mas vivo de los colores abundantes."""
    try:
        import mss
        import numpy as np
        with (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) as sct:
            img = np.asarray(sct.grab({"left": zona[0], "top": zona[1], "width": zona[2], "height": zona[3]}))
        rgb = img[:, :, :3][:, :, ::-1].reshape(-1, 3).astype(int)
        cuant = (rgb // 32) * 32 + 16
        colores, cuenta = np.unique(cuant, axis=0, return_counts=True)
        abundantes = colores[cuenta >= 0.05 * len(rgb)]
        if len(abundantes) == 0:
            abundantes = colores[[cuenta.argmax()]]
        viveza = abundantes.max(1) - abundantes.min(1) + abundantes.mean(1) * 0.2
        mejor = abundantes[viveza.argmax()]
        # media real de los pixeles de ese grupo (mas exacta que el centro del grupo)
        mascara = (cuant == mejor).all(1)
        return [int(v) for v in rgb[mascara].mean(0)]
    except Exception:
        return None


class MarcoDesplazable(ttk.Frame):
    """Marco con barra de desplazamiento (para pantallas pequenas)."""

    def __init__(self, padre):
        super().__init__(padre)
        self.lienzo = tk.Canvas(self, bg=C["fondo"], highlightthickness=0)
        barra = ttk.Scrollbar(self, orient="vertical", command=self.lienzo.yview)
        self.interior = ttk.Frame(self.lienzo)
        self.interior.bind("<Configure>", lambda e: self._ajustar())
        self.ventana = self.lienzo.create_window((0, 0), window=self.interior, anchor="nw")
        self.lienzo.bind("<Configure>", lambda e: (self.lienzo.itemconfigure(self.ventana, width=e.width),
                                                   self._ajustar()))
        self.lienzo.configure(yscrollcommand=barra.set)
        self.lienzo.pack(side="left", fill="both", expand=True)
        self.barra = barra
        self.bind_all("<MouseWheel>", self._rueda, add="+")
        self.bind_all("<Button-4>", lambda e: self._mover(-1), add="+")
        self.bind_all("<Button-5>", lambda e: self._mover(1), add="+")

    def _ajustar(self):
        self.lienzo.configure(scrollregion=self.lienzo.bbox("all"))
        cabe = self.interior.winfo_reqheight() <= self.lienzo.winfo_height()
        if cabe:
            self.barra.pack_forget()
            self.lienzo.yview_moveto(0)
        elif not self.barra.winfo_ismapped():
            self.barra.pack(side="right", fill="y")

    def _mover(self, n):
        if self.winfo_ismapped() and self.barra.winfo_ismapped():
            self.lienzo.yview_scroll(n, "units")

    def _rueda(self, e):
        # solo si el raton esta encima de este marco (no de la consola)
        w = self.winfo_containing(e.x_root, e.y_root)
        while w is not None and w is not self:
            w = w.master
        if w is self:
            self._mover(-1 if e.delta > 0 else 1)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IA local")
        ancho = min(1180, self.winfo_screenwidth() - 40)
        alto = min(900, self.winfo_screenheight() - 80)
        self.geometry(f"{ancho}x{alto}+20+10")
        self.minsize(860, 560)
        estilos(self)
        self.proceso = None
        self.cola = queue.Queue()
        self.paginas = {}
        self.botones_menu = {}

        # --- barra lateral ---
        lateral = ttk.Frame(self, style="Panel.TFrame", width=200)
        lateral.pack(side="left", fill="y")
        lateral.pack_propagate(False)
        ttk.Label(lateral, text="IA local", style="Logo.TLabel").pack(anchor="w", padx=18, pady=(20, 2))
        ttk.Label(lateral, text="juega y programa", style="Panel.TLabel",
                  foreground=C["suave"]).pack(anchor="w", padx=18, pady=(0, 18))

        # --- zona principal ---
        principal = ttk.Frame(self)
        principal.pack(side="left", fill="both", expand=True)
        cabecera = ttk.Frame(principal, padding=(20, 14, 20, 6))
        cabecera.pack(fill="x")
        self.titulo = ttk.Label(cabecera, text="", style="Titulo.TLabel")
        self.titulo.pack(side="left")
        ttk.Button(cabecera, text="Detectar GPU", command=lambda: self.ejecutar("dispositivo.py", [])
                   ).pack(side="right")
        self.dispositivo = tk.StringVar(value="auto")
        ttk.Combobox(cabecera, textvariable=self.dispositivo, values=DISPOSITIVOS, width=9,
                     state="readonly").pack(side="right", padx=6)
        ttk.Label(cabecera, text="Dispositivo", style="SuaveFondo.TLabel").pack(side="right")

        self.consola(principal)  # se empaqueta antes para que siempre quede visible abajo
        self.contenedor = ttk.Frame(principal, padding=(14, 0, 14, 0))
        self.contenedor.pack(fill="both", expand=True)

        paginas = [("pc", "Juego de PC", self.pagina_pc), ("clasicos", "Juegos clasicos", self.pagina_clasicos),
                   ("codigo", "Codigo", self.pagina_codigo), ("ayuda", "Ayuda", self.pagina_ayuda)]
        for clave, texto, crear in paginas:
            marco = MarcoDesplazable(self.contenedor)
            crear(marco.interior)
            self.paginas[clave] = (marco, texto)
            b = ttk.Button(lateral, text=texto, style="Menu.TButton", command=lambda k=clave: self.mostrar(k))
            b.pack(fill="x")
            self.botones_menu[clave] = b
        ttk.Label(lateral, text="F9 pausa · F10 para\n(en juegos de PC)", style="Panel.TLabel",
                  foreground=C["suave"], justify="left").pack(side="bottom", anchor="w", padx=18, pady=16)
        self.mostrar("pc")

        self.after(100, self.leer_cola)
        self.protocol("WM_DELETE_WINDOW", self.cerrar)

    def mostrar(self, clave):
        for k, (marco, _) in self.paginas.items():
            marco.pack_forget()
            self.botones_menu[k].configure(style="Menu.TButton")
        marco, texto = self.paginas[clave]
        marco.pack(fill="both", expand=True)
        self.botones_menu[clave].configure(style="MenuActivo.TButton")
        self.titulo.configure(text=texto)

    # ---------- piezas ----------
    def tarjeta(self, padre, numero, titulo, descripcion, fila, col, colspan=1):
        t = ttk.Frame(padre, style="Tarjeta.TFrame", padding=14)
        t.grid(row=fila, column=col, columnspan=colspan, sticky="nsew", padx=5, pady=5)
        cab = ttk.Frame(t, style="Tarjeta.TFrame")
        cab.pack(fill="x")
        if numero:
            ttk.Label(cab, text=str(numero), style="Num.TLabel").pack(side="left", padx=(0, 8))
        ttk.Label(cab, text=titulo, style="Paso.TLabel").pack(side="left")
        if descripcion:
            d = ttk.Label(t, text=descripcion, style="Suave.TLabel", justify="left", wraplength=300)
            d.pack(anchor="w", pady=(4, 8))
            # el texto se ajusta al ancho de la TARJETA (si no, el texto largo ensancha la columna)
            t.bind("<Configure>", lambda e: d.configure(wraplength=max(200, e.width - 32)), add="+")
        cuerpo = ttk.Frame(t, style="Tarjeta.TFrame")
        cuerpo.pack(fill="both", expand=True)
        return cuerpo

    def campo(self, padre, fila, etiqueta, valor="", tipo="texto", opciones=None, ancho=10, ayuda=None):
        ttk.Label(padre, text=etiqueta, style="Tarjeta.TLabel").grid(row=fila, column=0, sticky="w", pady=3,
                                                                    padx=(0, 10))
        if tipo == "check":
            var = tk.BooleanVar(value=valor)
            ttk.Checkbutton(padre, variable=var).grid(row=fila, column=1, sticky="w")
        else:
            var = tk.StringVar(value=str(valor))
            if opciones is not None:
                w = ttk.Combobox(padre, textvariable=var, values=opciones, width=ancho)
            else:
                w = ttk.Entry(padre, textvariable=var, width=ancho)
            w.grid(row=fila, column=1, sticky="w")
            if tipo in ("archivo", "carpeta"):
                def buscar():
                    r = filedialog.askdirectory(initialdir=RAIZ) if tipo == "carpeta" else \
                        filedialog.askopenfilename(initialdir=os.path.join(RAIZ, "modelos"))
                    if r:
                        var.set(r)
                ttk.Button(padre, text="...", width=3, command=buscar).grid(row=fila, column=2, sticky="w", padx=4)
        if ayuda:
            ttk.Label(padre, text=ayuda, style="Suave.TLabel").grid(row=fila, column=3, sticky="w", padx=6)
        return var

    # ---------- pagina: juego de PC ----------
    def pagina_pc(self, f):
        rejilla = ttk.Frame(f)
        rejilla.pack(fill="both", expand=True)
        for c in (0, 1):
            rejilla.columnconfigure(c, weight=1, minsize=0)

        # Perfil del juego
        p = self.tarjeta(rejilla, None, "Tu juego",
                         "Pon un nombre (se guarda todo con ese nombre) y marca la zona de la pantalla donde "
                         "esta el juego. Mejor con el juego en ventana.", 0, 0, colspan=2)
        ttk.Label(p, text="Nombre", style="Tarjeta.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.pc_nombre = tk.StringVar(value=(self.juegos_existentes() or ["Minecraft"])[0])
        self.combo_juegos = ttk.Combobox(p, textvariable=self.pc_nombre, values=self.juegos_existentes(), width=16)
        self.combo_juegos.grid(row=0, column=1, sticky="w")
        self.combo_juegos.bind("<<ComboboxSelected>>", lambda e: self.actualizar_perfil())
        self.combo_juegos.bind("<FocusOut>", lambda e: self.actualizar_perfil())
        ttk.Label(p, text="Zona", style="Tarjeta.TLabel").grid(row=0, column=2, sticky="w", padx=(16, 8))
        self.pc_region = tk.StringVar(value="")
        ttk.Entry(p, textvariable=self.pc_region, width=16).grid(row=0, column=3, sticky="w")
        ttk.Button(p, text="Elegir en pantalla", style="Acento.TButton",
                   command=lambda: elegir_zona(self, "Marca la ventana del juego",
                                               lambda z: self.pc_region.set(",".join(map(str, z))))
                   ).grid(row=0, column=4, padx=6)
        ttk.Button(p, text="Completa", command=lambda: self.pc_region.set("")).grid(row=0, column=5)
        self.perfil = ttk.Label(p, text="", style="Suave.TLabel", justify="left", wraplength=500)
        self.perfil.grid(row=1, column=0, columnspan=6, sticky="w", pady=(10, 0))
        p.master.bind("<Configure>", lambda e: self.perfil.configure(wraplength=max(300, e.width - 32)), add="+")

        # 1. Grabar
        g = self.tarjeta(rejilla, 1, "Grabar como juegas",
                         "Se graban TODAS las teclas que pulses, la camara y los clics. "
                         "F9 empieza/pausa, F10 termina.", 1, 0)
        fps = self.campo(g, 0, "Fotogramas/seg", 15, ayuda="10-20")
        raton = self.campo(g, 1, "Grabar raton", True, tipo="check")
        teclas = self.campo(g, 2, "Solo estas teclas", "", ancho=14, ayuda="vacio = todas")
        ttk.Button(g, text="Grabar", style="Acento.TButton", command=lambda: self.ejecutar("grabar_pc.py", [
            "--nombre", self.pc_nombre.get(), "--region", self.pc_region.get(), "--fps", fps.get(),
            "--teclas", teclas.get() or "todas"] + ([] if raton.get() else ["--sin-raton"]))
        ).grid(row=3, column=0, sticky="w", pady=(8, 0))

        # 2. Entrenar imitacion
        e = self.tarjeta(rejilla, 2, "Aprender de tus grabaciones",
                         "La IA aprende a hacer lo mismo que tu ante lo que ve en pantalla.", 1, 1)
        epocas = self.campo(e, 0, "Epocas", 15, ayuda="mas = mejor")
        ttk.Button(e, text="Entrenar", style="Acento.TButton", command=lambda: self.ejecutar("entrenar_pc.py", [
            "--nombre", self.pc_nombre.get(), "--epocas", epocas.get()])).grid(row=1, column=0, sticky="w",
                                                                                pady=(8, 0))

        # 3. Jugar
        j = self.tarjeta(rejilla, 3, "Que juegue imitandote",
                         "Tienes 5 segundos para hacer clic en el juego. F10 la para.", 2, 0)
        umbral = self.campo(j, 0, "Umbral teclas", 0.5, ayuda="baja si pulsa poco")
        escala = self.campo(j, 1, "Velocidad raton", 1.0, ayuda="sube si gira poco")
        suave = self.campo(j, 2, "Suavizado raton", 0.5, ayuda="0 a 0.9")
        ttk.Button(j, text="Jugar", style="Acento.TButton", command=lambda: self.ejecutar("jugar_pc.py", [
            "--modelo", os.path.join("modelos", f"pc_{self.pc_nombre.get()}.pt"), "--umbral", umbral.get(),
            "--raton-escala", escala.get(), "--suavizado", suave.get()]
            + (["--region", self.pc_region.get()] if self.pc_region.get() else []))
        ).grid(row=3, column=0, sticky="w", pady=(8, 0))

        # 4. Aprender sola
        a = self.tarjeta(rejilla, 4, "Que aprenda sola",
                         "Juega por su cuenta: explora por curiosidad, busca los objetos que le ensenes "
                         "(abajo) y, si marcas una, intenta llenar una barra (experiencia...).", 2, 1)
        minutos = self.campo(a, 0, "Minutos", 60)
        desde = self.campo(a, 1, "Partir de lo imitado", True, tipo="check")
        ttk.Label(a, text="Barra", style="Tarjeta.TLabel").grid(row=2, column=0, sticky="w", pady=3)
        barra = tk.StringVar(value="")
        ttk.Entry(a, textvariable=barra, width=12).grid(row=2, column=1, sticky="w")
        color = tk.StringVar(value="")
        muestra = tk.Label(a, text="   ", bg=C["tarjeta"], relief="flat", width=3)
        muestra.grid(row=3, column=2, sticky="w", padx=4)

        def poner_color(*_):
            try:
                r, g_, b = (int(v) for v in color.get().split(","))
                muestra.configure(bg=f"#{r:02x}{g_:02x}{b:02x}")
            except ValueError:
                muestra.configure(bg=C["tarjeta"])
        color.trace_add("write", poner_color)

        def barra_elegida(z):
            barra.set(",".join(map(str, z)))
            c = color_de_barra(z)
            if c:
                color.set(",".join(map(str, c)))

        ttk.Button(a, text="Elegir", width=6, command=lambda: elegir_zona(self, "Marca la barra (vida, experiencia...)",
                                                                 barra_elegida)).grid(row=2, column=2, padx=4)
        ttk.Label(a, text="Color", style="Tarjeta.TLabel").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(a, textvariable=color, width=12).grid(row=3, column=1, sticky="w")
        invertida = self.campo(a, 4, "Menos color = mejor", False, tipo="check")
        botones = ttk.Frame(a, style="Tarjeta.TFrame")
        botones.grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))

        def entrenar_sola():
            args = ["--nombre", self.pc_nombre.get(), "--region", self.pc_region.get(), "--minutos", minutos.get()]
            if barra.get().strip():
                args += ["--barra", barra.get(), "--barra-color", color.get() or "0,200,0"]
            args += (["--desde-imitacion"] if desde.get() else []) + (["--barra-invertida"] if invertida.get() else [])
            self.ejecutar("autoentrenar_pc.py", args)
        ttk.Button(botones, text="Entrenar sola", style="Acento.TButton", command=entrenar_sola).pack(side="left")
        ttk.Button(botones, text="Ver jugar", command=lambda: self.ejecutar("autoentrenar_pc.py", [
            "--nombre", self.pc_nombre.get(), "--solo-jugar", "--minutos", minutos.get()])).pack(side="left", padx=6)

        # Objetos
        o = self.tarjeta(rejilla, None, "Ensenar objetos (para que aprenda sola)",
                         "Pon el juego delante de un objeto (p. ej. un tronco), escribe su nombre y pulsa "
                         "'Marcar en pantalla': marca un trozo donde se vea SOLO ese objeto. Anade varios "
                         "ejemplos (de cerca, de lejos, con sombra). La IA buscara, se acercara y picara/recogera "
                         "esos objetos.", 3, 0, colspan=2)
        ttk.Label(o, text="Objeto", style="Tarjeta.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.objeto = tk.StringVar(value="madera")
        ttk.Combobox(o, textvariable=self.objeto, width=14, values=["madera", "piedra", "carbon", "hierro",
                                                                   "hojas", "enemigo", "comida"]
                     ).grid(row=0, column=1, sticky="w")
        ttk.Button(o, text="Marcar en pantalla", style="Acento.TButton",
                   command=lambda: elegir_zona(self, f"Marca un trozo de '{self.objeto.get()}' (solo el objeto)",
                                               self.objeto_marcado)).grid(row=0, column=2, padx=6)
        ttk.Button(o, text="Probar deteccion", command=lambda: self.ejecutar("ensenar_objeto.py", [
            "--nombre", self.pc_nombre.get(), "--probar", "--region", self.pc_region.get()])).grid(row=0, column=3)
        ttk.Button(o, text="Borrar objeto", style="Peligro.TButton", command=self.borrar_objeto
                   ).grid(row=0, column=4, padx=6)
        self.lista_objetos = ttk.Label(o, text="", style="Suave.TLabel")
        self.lista_objetos.grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))

        for fila in (1, 2):
            rejilla.rowconfigure(fila, weight=1)
        self.actualizar_perfil()

    def objeto_marcado(self, zona):
        from juegos.objetos import anadir_ejemplo, capturar_zona
        nombre, objeto = self.pc_nombre.get().strip(), self.objeto.get().strip()
        if not (nombre and objeto):
            return
        ejemplo = capturar_zona(zona)
        pantalla = capturar_zona((0, 0, self.winfo_screenwidth(), self.winfo_screenheight()))
        total = anadir_ejemplo(nombre, objeto, ejemplo, raiz=RAIZ, pantalla=pantalla)
        self.escribir(f"Ensenado '{objeto}' para {nombre}: {total} pixeles de ejemplo. "
                      "Anade mas ejemplos con distinta luz o distancia.\n", "ok")
        self.actualizar_perfil()

    def borrar_objeto(self):
        from juegos.objetos import borrar_objeto
        nombre, objeto = self.pc_nombre.get().strip(), self.objeto.get().strip()
        if messagebox.askyesno("Borrar", f"Borrar todo lo ensenado de '{objeto}' en {nombre}?"):
            borrar_objeto(nombre, objeto, raiz=RAIZ)
            self.actualizar_perfil()

    def juegos_existentes(self):
        nombres = {os.path.basename(p) for p in glob.glob(os.path.join(RAIZ, "datos_pc", "*")) if os.path.isdir(p)}
        for p in glob.glob(os.path.join(RAIZ, "modelos", "*.pt")):
            base = os.path.basename(p)[:-3]
            for pref in ("pc_", "auto_"):
                if base.startswith(pref):
                    nombres.add(base[len(pref):])
        return sorted(nombres)

    def actualizar_perfil(self):
        nombre = self.pc_nombre.get().strip()
        if not nombre:
            return
        partes = []
        archivos = glob.glob(os.path.join(RAIZ, "datos_pc", nombre, "*.npz"))
        if archivos:
            try:
                import numpy as np
                minutos, teclas = 0.0, set()
                for a in archivos:
                    with np.load(a) as d:
                        minutos += len(d["etiquetas"]) / float(d["fps"]) / 60
                        teclas.update(str(k) for k, n in zip(d["teclas"], d["etiquetas"].sum(0)) if n > 0)
                partes.append(f"Grabado: {len(archivos)} sesiones, {minutos:.1f} min · teclas usadas: "
                              + (", ".join(sorted(teclas)) or "ninguna"))
            except Exception:
                partes.append(f"Grabado: {len(archivos)} sesiones")
        else:
            partes.append("Grabado: nada todavia (paso 1)")
        for archivo, texto in ((f"pc_{nombre}.pt", "Imitacion"), (f"auto_{nombre}.pt", "Aprende sola")):
            ruta = os.path.join(RAIZ, "modelos", archivo)
            if os.path.exists(ruta):
                fecha = time.strftime("%d/%m %H:%M", time.localtime(os.path.getmtime(ruta)))
                partes.append(f"{texto}: entrenado ({fecha})")
            else:
                partes.append(f"{texto}: sin entrenar")
        self.perfil.configure(text="\n".join(partes))
        if hasattr(self, "lista_objetos"):
            try:
                from juegos.objetos import objetos_de
                objs = objetos_de(nombre, raiz=RAIZ)
            except Exception:
                objs = {}
            self.lista_objetos.configure(
                text=("Objetos ensenados: " + ", ".join(f"{k} ({int(v.sum())} px)" for k, v in objs.items()))
                if objs else "Todavia no le has ensenado ningun objeto para este juego.")
        self.combo_juegos.configure(values=self.juegos_existentes())

    # ---------- pagina: juegos clasicos ----------
    def pagina_clasicos(self, f):
        rejilla = ttk.Frame(f)
        rejilla.pack(fill="both", expand=True)
        for c in (0, 1):
            rejilla.columnconfigure(c, weight=1, minsize=0)
        t = self.tarjeta(rejilla, 1, "Entrenar", "Aprende sola probando y recibiendo recompensas. Snake viene "
                         "incluido; tambien juegos de Gymnasium.", 0, 0)
        juego = self.campo(t, 0, "Juego", "snake", opciones=["snake", "CartPole-v1", "Acrobot-v1", "LunarLander-v3",
                                                            "ALE/Pong-v5", "ALE/Breakout-v5"], ancho=16)
        atari = self.campo(t, 1, "Es de Atari", False, tipo="check")
        episodios = self.campo(t, 2, "Episodios", 1500)
        ttk.Button(t, text="Entrenar", style="Acento.TButton", command=lambda: self.ejecutar("entrenar_juego.py", [
            "--juego", juego.get(), "--episodios", episodios.get()] + (["--atari"] if atari.get() else []))
        ).grid(row=3, column=0, sticky="w", pady=(8, 0))
        t = self.tarjeta(rejilla, 2, "Ver jugar", "Snake se ve aqui abajo, en la consola.", 0, 1)
        modelo = self.campo(t, 0, "Modelo", "modelos/snake_mejor.pt", tipo="archivo", ancho=26)
        partidas = self.campo(t, 1, "Partidas", 3)
        ttk.Button(t, text="Jugar", style="Acento.TButton", command=lambda: self.ejecutar("jugar.py", [
            "--modelo", modelo.get(), "--partidas", partidas.get()])).grid(row=2, column=0, sticky="w", pady=(8, 0))

    # ---------- pagina: codigo ----------
    def pagina_codigo(self, f):
        rejilla = ttk.Frame(f)
        rejilla.pack(fill="both", expand=True)
        for c in (0, 1):
            rejilla.columnconfigure(c, weight=1, minsize=0)
        modelos = ["Qwen/Qwen2.5-Coder-0.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                   "Qwen/Qwen2.5-Coder-3B-Instruct"]
        t = self.tarjeta(rejilla, 1, "Asistente: ajustar con tus datos",
                         "Parte de un modelo que ya sabe programar y lo adapta a tu codigo y ejemplos .jsonl. "
                         "La primera vez descarga el modelo.", 0, 0)
        base = self.campo(t, 0, "Modelo base", modelos[0], opciones=modelos, ancho=30)
        datos = self.campo(t, 1, "Carpeta de codigo", "", tipo="carpeta", ancho=26)
        ejemplos = self.campo(t, 2, "Ejemplos .jsonl", "ejemplos/ejemplos.jsonl", tipo="archivo", ancho=26)
        pasos = self.campo(t, 3, "Pasos", 300)

        def ajustar():
            rutas = [r for r in (datos.get(), ejemplos.get()) if r.strip()]
            if not rutas:
                messagebox.showwarning("Faltan datos", "Elige una carpeta de codigo o un archivo .jsonl")
                return
            self.ejecutar("finetune_codigo.py", ["--modelo", base.get(), "--pasos", pasos.get(), "--datos", *rutas])
        ttk.Button(t, text="Ajustar", style="Acento.TButton", command=ajustar).grid(row=4, column=0, sticky="w",
                                                                                    pady=(8, 0))
        t = self.tarjeta(rejilla, 2, "Asistente: chatear",
                         "Escribe tus preguntas en la caja de abajo de la consola.", 0, 1)
        lora = self.campo(t, 0, "Adaptador LoRA", "modelos/lora_codigo", tipo="carpeta", ancho=26)
        sin_lora = self.campo(t, 1, "Modelo base sin ajustar", False, tipo="check")
        ttk.Button(t, text="Iniciar chat", style="Acento.TButton", command=lambda: self.ejecutar(
            "chat_codigo.py", ["--modelo", base.get()] if sin_lora.get() else ["--lora", lora.get()])
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

        t = self.tarjeta(rejilla, 3, "Mini GPT desde cero",
                         "Entrena un GPT pequeno solo con tus archivos. Para aprender como funciona por dentro.", 1, 0)
        carpeta = self.campo(t, 0, "Carpeta con codigo", "", tipo="carpeta", ancho=26)
        tamano = self.campo(t, 1, "Tamano", "pequeno", opciones=["pequeno", "mediano", "grande"])
        pasos_gpt = self.campo(t, 2, "Pasos", 5000)
        ttk.Button(t, text="Entrenar", style="Acento.TButton", command=lambda: self.ejecutar("entrenar_codigo.py", [
            "--datos", carpeta.get(), "--tamano", tamano.get(), "--pasos", pasos_gpt.get()], requiere=[carpeta])
        ).grid(row=3, column=0, sticky="w", pady=(8, 0))
        t = self.tarjeta(rejilla, 4, "Mini GPT: generar", None, 1, 1)
        prompt = self.campo(t, 0, "Empieza con", "def ", ancho=28)
        largo = self.campo(t, 1, "Largo (letras)", 500)
        ttk.Button(t, text="Generar", style="Acento.TButton", command=lambda: self.ejecutar("generar_codigo.py", [
            "--prompt", prompt.get(), "--largo", largo.get()])).grid(row=2, column=0, sticky="w", pady=(8, 0))

    # ---------- pagina: ayuda ----------
    def pagina_ayuda(self, f):
        t = self.tarjeta(f, None, "Como empezar", None, 0, 0)
        f.columnconfigure(0, weight=1)
        texto = (
            "Juego de PC (cualquier juego):\n"
            "  1. Pon el nombre del juego y pulsa 'Elegir en pantalla' para marcar la ventana del juego.\n"
            "  2. Grabar: juega normal 30-60 min (F9 empieza/pausa, F10 termina). Graba todas las teclas y el raton.\n"
            "  3. Entrenar, y despues Jugar: la IA te imita.\n"
            "  4. Que aprenda sola: sigue mejorando por su cuenta. Marca una barra (experiencia, vida) para\n"
            "     darle un objetivo. Necesita horas.\n\n"
            "GPU integrada: 'auto' la usa si la encuentra. Pulsa 'Detectar GPU' para ver cual usa.\n"
            "  Intel: pip install torch --index-url https://download.pytorch.org/whl/xpu\n"
            "  AMD en Windows: pip install torch-directml\n\n"
            "Seguridad: la IA nunca pulsa Esc, F4, F9, F10 ni la tecla de Windows.\n"
            "No la uses en juegos online con anti-trampas: te pueden banear.")
        ttk.Label(t, text=texto, style="Tarjeta.TLabel", justify="left").pack(anchor="w")

    # ---------- consola ----------
    def consola(self, padre):
        marco = ttk.Frame(padre, padding=(20, 6, 20, 14))
        marco.pack(fill="x", side="bottom")
        barra = ttk.Frame(marco)
        barra.pack(fill="x", pady=(0, 6))
        self.punto = tk.Label(barra, text="●", fg=C["suave"], bg=C["fondo"], font=(FUENTE[0], 12))
        self.punto.pack(side="left")
        self.estado = tk.StringVar(value="Listo")
        ttk.Label(barra, textvariable=self.estado).pack(side="left", padx=6)
        self.progreso = ttk.Progressbar(barra, mode="indeterminate", length=160)
        ttk.Button(barra, text="Limpiar", command=lambda: self.log.delete("1.0", "end")).pack(side="right")
        ttk.Button(barra, text="Detener", style="Peligro.TButton", command=self.detener).pack(side="right", padx=6)
        caja = tk.Frame(marco, bg=C["borde"], padx=1, pady=1)
        caja.pack(fill="both", expand=True)
        self.log = tk.Text(caja, height=8, font=MONO, wrap="word", bg=C["consola"], fg="#cfd3dc",
                           insertbackground=C["texto"], relief="flat", padx=10, pady=8, borderwidth=0,
                           highlightthickness=0)
        scroll = ttk.Scrollbar(caja, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_configure("cmd", foreground=C["acento"])
        self.log.tag_configure("error", foreground=C["error"])
        self.log.tag_configure("ok", foreground=C["ok"])
        self.log.tag_configure("aviso", foreground=C["aviso"])
        entrada = ttk.Frame(marco)
        entrada.pack(fill="x", pady=(6, 0))
        self.texto = tk.StringVar()
        caja_texto = ttk.Entry(entrada, textvariable=self.texto)
        caja_texto.pack(side="left", fill="x", expand=True)
        caja_texto.bind("<Return>", lambda e: self.enviar())
        ttk.Button(entrada, text="Enviar", command=self.enviar).pack(side="left", padx=(6, 0))

    def poner_estado(self, texto, color, ocupado):
        self.estado.set(texto)
        self.punto.configure(fg=color)
        if ocupado:
            self.progreso.pack(side="left", padx=10)
            self.progreso.start(12)
        else:
            self.progreso.stop()
            self.progreso.pack_forget()

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
        self.escribir(f"\n$ python {' '.join(cmd[2:])}\n", "cmd")
        entorno = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.proceso = subprocess.Popen(cmd, cwd=RAIZ, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, env=entorno, bufsize=0)
        self.poner_estado(f"En marcha: {script}", C["aviso"], True)
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
                    if item[1] == 0:
                        self.poner_estado("Listo", C["ok"], False)
                    else:
                        self.poner_estado(f"Terminado con error (codigo {item[1]})", C["error"], False)
                    self.actualizar_perfil()
                else:
                    self.escribir(item)
        except queue.Empty:
            pass
        self.after(50, self.leer_cola)

    def escribir(self, texto, etiqueta=None):
        # \f lo usa jugar.py para "limpiar pantalla" y animar Snake
        if "\f" in texto:
            self.log.delete("1.0", "end")
            self.log.insert("end", texto.rsplit("\f", 1)[1])
            self.log.see("1.0")
            return
        if etiqueta is None:
            for linea in texto.splitlines(keepends=True):
                baja = linea.lower()
                tag = ("error" if ("error" in baja or "traceback" in baja) else
                       "aviso" if "aviso" in baja else
                       "ok" if any(p in baja for p in ("listo", "guardado", "grabando", "jugando")) else None)
                self.log.insert("end", linea, tag or ())
        else:
            self.log.insert("end", texto, etiqueta)
        self.log.see("end")

    def enviar(self):
        texto = self.texto.get()
        if not (self.proceso and self.proceso.poll() is None):
            return
        self.escribir(texto + "\n", "cmd")
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
            self.escribir("\n[Detenido]\n", "aviso")

    def cerrar(self):
        self.detener()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
