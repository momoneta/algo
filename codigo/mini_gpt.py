"""Mini GPT (transformer decodificador) a nivel de bytes.

Trabaja con bytes (0-255), asi que entiende cualquier lenguaje de programacion
y cualquier texto sin necesitar un tokenizador.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TAMANOS = {
    # nombre: (capas, cabezas, dim, contexto)  -> elige segun tu GPU/RAM
    "pequeno": (4, 4, 256, 256),     # ~3M parametros, va bien en GPU integrada o CPU
    "mediano": (6, 6, 384, 512),     # ~11M parametros
    "grande": (8, 8, 512, 768),      # ~25M parametros, mejor con GPU dedicada
}


class Bloque(nn.Module):
    def __init__(self, dim, cabezas, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proy = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))
        self.cabezas = cabezas
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.cabezas, C // self.cabezas).transpose(1, 2) for t in (q, k, v))
        # Atencion causal hecha a mano (compatible con CPU, CUDA, XPU, MPS y DirectML)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        mascara = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mascara, float("-inf"))
        att = F.dropout(F.softmax(att, dim=-1), self.dropout, self.training)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proy(y)
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, capas=4, cabezas=4, dim=256, contexto=256, dropout=0.1, vocab=256):
        super().__init__()
        self.config = dict(capas=capas, cabezas=cabezas, dim=dim, contexto=contexto, dropout=dropout, vocab=vocab)
        self.contexto = contexto
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(contexto, dim)
        self.bloques = nn.ModuleList([Bloque(dim, cabezas, dropout) for _ in range(capas)])
        self.ln = nn.LayerNorm(dim)
        self.salida = nn.Linear(dim, vocab, bias=False)
        self.salida.weight = self.tok.weight  # pesos compartidos
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, objetivos=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.bloques:
            x = b(x)
        logits = self.salida(self.ln(x))
        perdida = None
        if objetivos is not None:
            perdida = F.cross_entropy(logits.view(-1, logits.size(-1)), objetivos.view(-1))
        return logits, perdida

    @torch.no_grad()
    def generar(self, idx, n_nuevos, temperatura=0.8, top_k=40):
        for _ in range(n_nuevos):
            logits, _ = self(idx[:, -self.contexto:])
            logits = logits[:, -1, :] / max(temperatura, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            sig = torch.multinomial(F.softmax(logits, dim=-1).cpu(), 1).to(idx.device)
            idx = torch.cat([idx, sig], dim=1)
        return idx

    def n_parametros(self):
        return sum(p.numel() for p in self.parameters())
