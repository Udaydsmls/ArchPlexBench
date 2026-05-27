import argparse
import os

import torch
import torch.nn as nn
import wandb
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import get_dataloaders, get_tokenizer
from evaluate import compute_perplexity
from models import build_model


def _train_epoch(model, loader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_model(cfg, vocab_size, train_loader, val_loader, device, epochs=20, lr=3e-4):
    """Train a single model and save the best checkpoint. Returns (model, best_val_ppl)."""
    model = build_model(cfg, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{cfg['model']}_best.pt"
    best_ppl = float("inf")

    width = cfg.get("d_model", cfg.get("embed_dim", "na"))
    wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    wandb.init(
        project="archplexbench",
        name=f"{cfg['model']}_{width}d",
        config=cfg,
        mode=wandb_mode,
        reinit=True,
    )

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm=1.0)
        val_ppl = compute_perplexity(model, val_loader, device)
        scheduler.step()
        print(f"  Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | val_ppl={val_ppl:.2f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_ppl": val_ppl})

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save(model.state_dict(), ckpt_path)

    wandb.finish()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, best_ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = get_dataloaders(args.seq_len, args.batch_size, tokenizer)

    model, val_ppl = train_model(cfg, tokenizer.vocab_size, train_loader, val_loader, device, args.epochs, args.lr)

    from evaluate import compute_perplexity
    test_ppl = compute_perplexity(model, test_loader, device)
    print(f"\nBest val PPL: {val_ppl:.2f} | Test PPL: {test_ppl:.2f}")
