import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
import yaml

from data.dataset import get_dataloaders, get_tokenizer
from evaluate import compute_perplexity
from train import train_model

CONFIGS = [
    "configs/transformer_decoder.yaml",
    "configs/encoder_decoder.yaml",
    "configs/lstm.yaml",
    "configs/mamba.yaml",
    "configs/mamba_multihead.yaml",
]


def _plot(df):
    colors = list(plt.cm.tab10.colors[: len(df)])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(df["model"], df["test_ppl"], color=colors)
    axes[0].set_title("Test Perplexity (lower is better)")
    axes[0].set_ylabel("Perplexity")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(df["model"], df["params_M"], color=colors)
    axes[1].set_title("Trainable Parameters (M)")
    axes[1].set_ylabel("Millions")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/comparison.png", dpi=150)
    plt.show()


def run_benchmark(epochs=20, lr=3e-4, batch_size=32, seq_len=256):
    """Train all architectures and compare test perplexity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    train_loader, val_loader, test_loader = get_dataloaders(seq_len, batch_size, tokenizer)

    results = []
    for config_path in CONFIGS:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        print(f"{'='*50}")
        print(f"Model: {cfg['model'].upper()}")
        try:
            model, val_ppl = train_model(cfg, vocab_size, train_loader, val_loader, device, epochs, lr)

            test_ppl = compute_perplexity(model, test_loader, device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            results.append({
                "model": cfg["model"],
                "val_ppl": round(val_ppl, 2),
                "test_ppl": round(test_ppl, 2),
                "params_M": round(n_params / 1e6, 2),
            })
            print(f"  Test PPL: {test_ppl:.2f}\n")
        finally:
            if wandb.run is not None:
                wandb.finish()

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/benchmark.csv", index=False)

    print("\n" + "=" * 50)
    print(df.to_string(index=False))

    _plot(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    args = parser.parse_args()

    run_benchmark(args.epochs, args.lr, args.batch_size, args.seq_len)
