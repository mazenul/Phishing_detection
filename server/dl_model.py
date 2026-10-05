"""
DL inference module — loads the character-level Transformer and exposes predict_dl().
Architecture is identical to DL/phishing_transformer.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent          # folder_final try/
_VOCAB_PATH = _ROOT / "DL" / "vocab.json"
_MODEL_PATH = _ROOT / "DL" / "best_model.pt"

# ── Hyper-parameters (must match training) ────────────────────────────────────
MAX_LEN    = 200
EMBED_DIM  = 64
NUM_HEADS  = 4
FF_DIM     = 128
NUM_BLOCKS = 2
DROPOUT    = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model architecture ────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class PhishingTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, embed_dim: int,
                 num_heads: int, ff_dim: int, num_blocks: int, dropout: float):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim, max_len, dropout)
        self.blocks  = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


# ── Module-level state ────────────────────────────────────────────────────────
_vocab: dict[str, int] | None = None
_model: PhishingTransformer | None = None


def _load() -> None:
    global _vocab, _model

    with open(_VOCAB_PATH, encoding="utf-8") as f:
        _vocab = json.load(f)

    vocab_size = len(_vocab)
    _model = PhishingTransformer(
        vocab_size, MAX_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_BLOCKS, DROPOUT
    ).to(DEVICE)
    state = torch.load(_MODEL_PATH, map_location=DEVICE, weights_only=True)
    _model.load_state_dict(state)
    _model.eval()
    print(f"[DL] Transformer loaded — vocab={vocab_size}, device={DEVICE}")


def _encode(url: str) -> np.ndarray:
    chars   = list(url[:MAX_LEN])
    indices = [_vocab.get(c, 1) for c in chars]   # 1 = <UNK>
    padded  = indices + [0] * (MAX_LEN - len(indices))
    return np.array(padded, dtype=np.int32)


# ── Public API ────────────────────────────────────────────────────────────────

def load_dl_model() -> None:
    """Call once at server startup."""
    _load()


@torch.no_grad()
def predict_dl(url: str) -> float:
    """Return phishing probability in [0, 1]."""
    encoded = _encode(url)
    tensor  = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(DEVICE)
    logit   = _model(tensor)
    prob    = torch.sigmoid(logit).item()
    return float(prob)
