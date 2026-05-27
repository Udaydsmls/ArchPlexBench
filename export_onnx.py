"""Export a trained Mamba checkpoint to ONNX.

Usage:
    python export_onnx.py \
        --checkpoint checkpoints/mamba_best.pt \
        --config configs/mamba.yaml \
        --output mamba.onnx

The exported graph takes integer token ids (batch, seq_len) and returns
logits (batch, seq_len, vocab_size). Batch and sequence length are dynamic
so the same file works for any input size at inference time.
"""

import argparse

import torch
import yaml

from data.dataset import get_tokenizer
from models import build_model


def export(checkpoint: str, config: str, output: str) -> None:
    device = torch.device("cpu")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    tokenizer = get_tokenizer()
    model = build_model(cfg, tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Minimal dummy input — just needs to be a valid token id tensor
    dummy = torch.zeros(1, 16, dtype=torch.long)

    torch.onnx.export(
        model,
        dummy,
        output,
        opset_version=17,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/mamba.yaml")
    parser.add_argument("--output", default="mamba.onnx")
    args = parser.parse_args()

    export(args.checkpoint, args.config, args.output)
