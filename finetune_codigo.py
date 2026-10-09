"""Ajusta (fine-tuning con LoRA) un modelo de codigo preentrenado con TUS datos.

Esto da resultados mucho mejores que entrenar desde cero, porque el modelo ya
sabe programar y solo aprende tu estilo / tus proyectos / tus ejemplos.

Datos admitidos (se pueden mezclar):
  * carpetas con codigo -> aprende a continuar tu codigo
  * archivos .jsonl con lineas {"pregunta": "...", "respuesta": "..."} -> aprende a responder

  python finetune_codigo.py --datos ./mi_codigo --pasos 300
  python finetune_codigo.py --datos ejemplos.jsonl --modelo Qwen/Qwen2.5-Coder-1.5B-Instruct

Luego:  python chat_codigo.py --lora modelos/lora_codigo
"""
import argparse
import json
import os
import random
import time

import torch

from dispositivo import elegir_dispositivo, describir
from entrenar_codigo import EXT_POR_DEFECTO, CARPETAS_IGNORADAS


def cargar_ejemplos(rutas, extensiones, tokenizer):
    ejemplos = []
    for ruta in rutas:
        if ruta.endswith(".jsonl"):
            with open(ruta, encoding="utf-8") as f:
                for linea in f:
                    if linea.strip():
                        e = json.loads(linea)
                        msgs = [{"role": "user", "content": e["pregunta"]},
                                {"role": "assistant", "content": e["respuesta"]}]
                        ejemplos.append(tokenizer.apply_chat_template(msgs, tokenize=False))
            continue
        archivos = [ruta] if os.path.isfile(ruta) else []
        for raiz, dirs, nombres in os.walk(ruta):
            dirs[:] = [d for d in dirs if d not in CARPETAS_IGNORADAS]
            archivos += [os.path.join(raiz, n) for n in nombres if n.lower().endswith(extensiones)]
        for a in archivos:
            try:
                with open(a, encoding="utf-8") as f:
                    ejemplos.append(f"# Archivo: {os.path.basename(a)}\n" + f.read())
            except (OSError, UnicodeDecodeError):
                pass
    return ejemplos


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", nargs="+", required=True)
    p.add_argument("--modelo", default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                   help="modelo base de HuggingFace (0.5B va bien en GPU integrada)")
    p.add_argument("--extensiones", default=EXT_POR_DEFECTO)
    p.add_argument("--pasos", type=int, default=300)
    p.add_argument("--lote", type=int, default=1)
    p.add_argument("--acumular", type=int, default=8, help="acumulacion de gradiente (lote efectivo = lote*acumular)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--contexto", type=int, default=512)
    p.add_argument("--rango-lora", type=int, default=16)
    p.add_argument("--dispositivo", default="auto", help="auto|cuda|xpu|mps|directml|cpu")
    p.add_argument("--salida", default="modelos/lora_codigo")
    args = p.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dispositivo = elegir_dispositivo(args.dispositivo)
    tipo = getattr(dispositivo, "type", "cpu")
    print(f"Dispositivo: {describir(dispositivo)}")

    tokenizer = AutoTokenizer.from_pretrained(args.modelo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    exts = tuple(e.strip().lower() for e in args.extensiones.split(",") if e.strip())
    textos = cargar_ejemplos(args.datos, exts, tokenizer)
    if not textos:
        raise SystemExit("No se encontraron datos.")

    # Trocear en bloques del tamano del contexto
    bloques = []
    for t in textos:
        ids = tokenizer(t + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        for i in range(0, len(ids), args.contexto):
            trozo = ids[i:i + args.contexto]
            if len(trozo) > 16:
                bloques.append(trozo)
    print(f"Datos: {len(textos)} documentos -> {len(bloques)} bloques de hasta {args.contexto} tokens")

    # bfloat16 en GPUs que lo soportan; float32 en CPU/DirectML por compatibilidad
    dtype = torch.bfloat16 if tipo in ("cuda", "xpu") else torch.float32
    modelo = AutoModelForCausalLM.from_pretrained(args.modelo, torch_dtype=dtype)
    modelo.gradient_checkpointing_enable()
    modelo.enable_input_require_grads()
    lora = LoraConfig(r=args.rango_lora, lora_alpha=args.rango_lora * 2, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    modelo = get_peft_model(modelo, lora)
    modelo.print_trainable_parameters()
    modelo.to(dispositivo)
    modelo.train()

    optim = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=args.lr)
    inicio = time.time()
    for paso in range(1, args.pasos + 1):
        total = 0.0
        for _ in range(args.acumular):
            lote = random.sample(bloques, min(args.lote, len(bloques)))
            largo = max(len(b) for b in lote)
            ids = torch.full((len(lote), largo), tokenizer.pad_token_id)
            mascara = torch.zeros((len(lote), largo), dtype=torch.long)
            for i, b in enumerate(lote):
                ids[i, :len(b)] = torch.tensor(b)
                mascara[i, :len(b)] = 1
            etiquetas = ids.masked_fill(mascara == 0, -100)
            salida = modelo(input_ids=ids.to(dispositivo), attention_mask=mascara.to(dispositivo),
                            labels=etiquetas.to(dispositivo))
            (salida.loss / args.acumular).backward()
            total += salida.loss.item() / args.acumular
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        for g in optim.param_groups:
            g["lr"] = args.lr * min(1.0, paso / 20)
        optim.step()
        optim.zero_grad(set_to_none=True)
        if paso % 10 == 0 or paso == 1:
            print(f"paso {paso:5d}/{args.pasos} | perdida {total:.3f} | {time.time() - inicio:6.0f}s")
        if paso % 100 == 0:
            modelo.save_pretrained(args.salida)

    modelo.save_pretrained(args.salida)
    tokenizer.save_pretrained(args.salida)
    with open(os.path.join(args.salida, "modelo_base.txt"), "w") as f:
        f.write(args.modelo)
    print(f"\nListo. Adaptador LoRA guardado en {args.salida}")
    print(f"Chatea con el: python chat_codigo.py --lora {args.salida}")


if __name__ == "__main__":
    main()
