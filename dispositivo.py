"""Deteccion del dispositivo de computo, incluidas GPUs integradas.

Orden automatico:
  1. cuda     -> GPU NVIDIA (o AMD con ROCm en Linux)
  2. xpu      -> GPU Intel (integrada Iris Xe / Arc, o dedicada Arc). PyTorch >= 2.5
  3. mps      -> GPU de Apple (M1/M2/M3...)
  4. directml -> cualquier GPU en Windows (integrada AMD Radeon o Intel), con torch-directml
  5. cpu

Se puede forzar con --dispositivo cuda|xpu|mps|directml|cpu
"""
import torch


def _directml():
    try:
        import torch_directml  # pip install torch-directml (solo Windows)
        return torch_directml.device()
    except Exception:
        return None


def elegir_dispositivo(preferido="auto"):
    preferido = (preferido or "auto").lower()

    if preferido == "cpu":
        return torch.device("cpu")
    if preferido == "directml":
        d = _directml()
        if d is None:
            raise RuntimeError("DirectML no disponible. Instala con: pip install torch-directml")
        return d
    if preferido != "auto":
        return torch.device(preferido)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    d = _directml()
    if d is not None:
        return d
    return torch.device("cpu")


def describir(dispositivo):
    tipo = getattr(dispositivo, "type", str(dispositivo))
    try:
        if tipo == "cuda":
            return f"cuda ({torch.cuda.get_device_name(0)})"
        if tipo == "xpu":
            return f"xpu ({torch.xpu.get_device_name(0)})"
    except Exception:
        pass
    if tipo == "privateuseone":
        return "directml (GPU via DirectML)"
    return str(tipo)


if __name__ == "__main__":
    d = elegir_dispositivo()
    print("Dispositivo detectado:", describir(d))
    x = torch.randn(512, 512, device=d)
    print("Prueba de multiplicacion de matrices OK:", (x @ x).shape)
