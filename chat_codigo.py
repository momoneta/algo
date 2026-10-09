"""Chat local con un modelo de codigo (con o sin tu ajuste LoRA).

  python chat_codigo.py                              # modelo base sin entrenar
  python chat_codigo.py --lora modelos/lora_codigo   # con tu entrenamiento
"""
import argparse
import os

import torch

from dispositivo import elegir_dispositivo, describir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default=None, help="modelo base (por defecto el usado al entrenar)")
    p.add_argument("--lora", default=None, help="carpeta del adaptador LoRA")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperatura", type=float, default=0.3)
    p.add_argument("--dispositivo", default="auto")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = args.modelo
    if base is None and args.lora and os.path.exists(os.path.join(args.lora, "modelo_base.txt")):
        base = open(os.path.join(args.lora, "modelo_base.txt")).read().strip()
    base = base or "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    d = elegir_dispositivo(args.dispositivo)
    tipo = getattr(d, "type", "cpu")
    print(f"Cargando {base} en {describir(d)}...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora or base)
    dtype = torch.bfloat16 if tipo in ("cuda", "xpu") else (torch.float16 if tipo == "mps" else torch.float32)
    modelo = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    if args.lora:
        from peft import PeftModel
        modelo = PeftModel.from_pretrained(modelo, args.lora)
        modelo = modelo.merge_and_unload()
    modelo.to(d).eval()

    historial = [{"role": "system", "content": "Eres un asistente experto en programacion. Responde en espanol."}]
    print("Escribe tu pregunta ('salir' para terminar, 'nuevo' para borrar el historial).")
    while True:
        try:
            pregunta = input("\nTu: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        if pregunta.lower() == "nuevo":
            historial = historial[:1]
            continue
        if not pregunta:
            continue
        historial.append({"role": "user", "content": pregunta})
        entrada = tokenizer.apply_chat_template(historial, add_generation_prompt=True, return_tensors="pt",
                                                return_dict=True).to(d)
        largo = entrada["input_ids"].shape[1]
        with torch.no_grad():
            salida = modelo.generate(**entrada, max_new_tokens=args.max_tokens, do_sample=args.temperatura > 0,
                                     temperature=max(args.temperatura, 1e-5), top_p=0.95,
                                     pad_token_id=tokenizer.eos_token_id)
        respuesta = tokenizer.decode(salida[0, largo:], skip_special_tokens=True)
        historial.append({"role": "assistant", "content": respuesta})
        print(f"\nIA: {respuesta}")


if __name__ == "__main__":
    main()
