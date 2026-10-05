# -*- coding: utf-8 -*-
"""
Character-level Transformer model for phishing URL detection.
Framework : PyTorch (CUDA GPU)
Dataset   : url_dataset_cleaned.csv  (columns: url, type, label)
Label     : 0 = legitimate,  1 = phishing
"""

import os
import sys
import csv
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, precision_score,
                              recall_score, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ config ---
DATA_PATH  = r"C:\Users\imaze\Desktop\Model traing DeepLearning using another dataset\cleaned work\url_dataset_cleaned.csv"
SAVE_DIR   = r"C:\Users\imaze\Desktop\Model traing DeepLearning using another dataset\cleaned work"

MAX_LEN    = 200
EMBED_DIM  = 64
NUM_HEADS  = 4
FF_DIM     = 128
NUM_BLOCKS = 2
DROPOUT    = 0.1
BATCH_SIZE = 512
EPOCHS     = 15
LR         = 1e-3
LOG_EVERY  = 50      # print progress every N batches

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg):
    print(msg, flush=True)

log(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    log(f"GPU : {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ----------------------------------------------------------------- helpers ---

def build_vocab(urls):
    chars = set()
    for u in urls:
        chars.update(u)
    vocab = {c: i + 2 for i, c in enumerate(sorted(chars))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


def encode_fast(urls, vocab, max_len):
    """Vectorised character encoding using numpy."""
    n = len(urls)
    out = np.zeros((n, max_len), dtype=np.int32)
    for i, url in enumerate(urls):
        chars = list(url[:max_len])
        indices = [vocab.get(c, 1) for c in chars]
        out[i, :len(indices)] = indices
    return out


# ------------------------------------------------------------------ dataset --

class URLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------------------------------- transformer model ---

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
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

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class PhishingTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim,
                 num_heads, ff_dim, num_blocks, dropout):
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

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


# -------------------------------------------------------------------- train ---

def train_epoch(model, loader, criterion, optimizer, scaler, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    t0 = time.time()

    for step, (X_batch, y_batch) in enumerate(loader, 1):
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = len(y_batch)
        total_loss += loss.item() * bs
        preds       = (torch.sigmoid(logits) >= 0.5).long()
        correct    += (preds == y_batch.long()).sum().item()
        total      += bs

        if step % LOG_EVERY == 0 or step == n_batches:
            elapsed = time.time() - t0
            eta     = elapsed / step * (n_batches - step)
            log(f"  Epoch {epoch}/{total_epochs} | "
                f"step {step}/{n_batches} | "
                f"loss {total_loss/total:.4f} | "
                f"acc {correct/total*100:.2f}% | "
                f"ETA {eta:.0f}s")

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type="cuda"):
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
        bs = len(y_batch)
        total_loss += loss.item() * bs
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        correct    += (preds == y_batch.long()).sum().item()
        total      += bs
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    return (total_loss / total, correct / total,
            np.array(all_probs), np.array(all_labels))


# -------------------------------------------------------------------  main ---

def main():
    log("=" * 60)
    log("  Phishing URL Detection - Character Transformer (GPU)")
    log("=" * 60)

    # load
    log("\n[1/6] Loading dataset ...")
    urls, labels = [], []
    with open(DATA_PATH, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            url = row["url"].strip()
            if url:
                urls.append(url)
                labels.append(int(row["label"].strip()))

    urls   = np.array(urls)
    labels = np.array(labels, dtype=np.float32)
    log(f"   Total    : {len(urls):,}")
    log(f"   Legit (0): {(labels==0).sum():,}")
    log(f"   Phish (1): {(labels==1).sum():,}")

    # vocab
    log("\n[2/6] Building vocabulary ...")
    vocab      = build_vocab(urls)
    vocab_size = len(vocab)
    log(f"   Vocab size: {vocab_size}")

    # encode
    log("\n[3/6] Encoding URLs ...")
    t0 = time.time()
    X  = encode_fast(urls, vocab, MAX_LEN)
    log(f"   Done in {time.time()-t0:.1f}s")

    # split
    log("\n[4/6] Splitting data ...")
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, labels, test_size=0.20, random_state=42, stratify=labels)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
    log(f"   Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_te):,}")

    train_loader = DataLoader(URLDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(URLDataset(X_val, y_val), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(URLDataset(X_te, y_te), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    # model
    log("\n[5/6] Building model ...")
    model = PhishingTransformer(vocab_size, MAX_LEN, EMBED_DIM,
                                NUM_HEADS, FF_DIM, NUM_BLOCKS, DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"   Trainable parameters: {total_params:,}")

    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    log(f"   pos_weight: {pos_w.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5)
    scaler    = torch.amp.GradScaler()

    # train
    log("\n[6/6] Training ...")
    best_val_auc, patience_cnt = 0.0, 0
    PATIENCE  = 3
    best_path = os.path.join(SAVE_DIR, "best_model.pt")
    history   = {"train_loss": [], "val_loss": [],
                 "train_acc":  [], "val_acc":  [], "val_auc": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, EPOCHS)
        val_loss, val_acc, val_probs, val_labels = eval_epoch(
            model, val_loader, criterion)
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_auc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        lr_now = optimizer.param_groups[0]["lr"]
        log(f"\n  === Epoch {epoch}/{EPOCHS} done in {time.time()-t0:.1f}s ===")
        log(f"  Train loss={tr_loss:.4f}  acc={tr_acc*100:.2f}%")
        log(f"  Val   loss={val_loss:.4f}  acc={val_acc*100:.2f}%  "
            f"auc={val_auc:.4f}  lr={lr_now:.6f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
            log(f"  -> Best model saved! (val_auc={best_val_auc:.4f})")
        else:
            patience_cnt += 1
            log(f"  -> No improvement ({patience_cnt}/{PATIENCE})")
            if patience_cnt >= PATIENCE:
                log(f"  Early stopping at epoch {epoch}.")
                break

    # test evaluation
    log("\n--- Test-set Evaluation ---")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    _, test_acc, test_probs, test_labels = eval_epoch(model, test_loader, criterion)
    test_preds = (test_probs >= 0.5).astype(int)

    auc  = roc_auc_score(test_labels, test_probs)
    prec = precision_score(test_labels, test_preds)
    rec  = recall_score(test_labels, test_preds)
    f1   = f1_score(test_labels, test_preds)
    log(f"  Accuracy  : {test_acc*100:.2f}%")
    log(f"  AUC       : {auc:.4f}")
    log(f"  Precision : {prec:.4f}")
    log(f"  Recall    : {rec:.4f}")
    log(f"  F1-score  : {f1:.4f}")
    log("\nClassification Report:")
    log(classification_report(test_labels, test_preds,
                              target_names=["Legitimate", "Phishing"]))
    log("Confusion Matrix:")
    log(str(confusion_matrix(test_labels, test_preds)))

    # plots
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(ep, history["train_acc"], label="Train")
    axes[0].plot(ep, history["val_acc"],   label="Val")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(ep, history["train_loss"], label="Train")
    axes[1].plot(ep, history["val_loss"],   label="Val")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    axes[2].plot(ep, history["val_auc"], label="Val AUC", color="green")
    axes[2].set_title("AUC"); axes[2].legend(); axes[2].set_xlabel("Epoch")
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    log(f"\nTraining curves saved -> {plot_path}")

    # save vocab
    vocab_path = os.path.join(SAVE_DIR, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    log(f"Vocabulary saved      -> {vocab_path}")
    log(f"Best model saved      -> {best_path}")

    # inference demo
    log("\n--- Quick Inference Demo ---")
    demo_urls = [
        "https://www.google.com",
        "https://paypa1-secure-login.com/verify?user=admin",
        "https://www.amazon.com/gp/cart",
        "http://192.168.1.1/phish/steal-credentials.php",
    ]
    demo_enc = torch.tensor(encode_fast(np.array(demo_urls), vocab, MAX_LEN),
                            dtype=torch.long).to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(demo_enc)).cpu().numpy()
    for url, p in zip(demo_urls, probs):
        tag = "PHISHING" if p >= 0.5 else "Legitimate"
        log(f"  [{tag:10s}] ({p:.3f})  {url}")


if __name__ == "__main__":
    main()
