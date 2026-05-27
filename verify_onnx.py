"""Check that an exported ONNX model matches the PyTorch original.

Usage:
    python verify_onnx.py \
        --onnx mamba.onnx \
        --checkpoint checkpoints/mamba_best.pt \
        --config configs/mamba.yaml

Runs the same random input through both runtimes and checks the max
absolute difference. Exits with code 1 if the check fails, so it can
be used as a CI gate.
"""

import argparse
import sys

import numpy as np
import onnxruntime as ort
import torch
import yaml

from data.dataset import get_tokenizer
from models import build_model


def verify(onnx_path: str, checkpoint: str, config: str, atol: float = 1e-4) -> bool:
    device = torch.device("cpu")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    tokenizer = get_tokenizer()
    model = build_model(cfg, tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    rng = torch.Generator()
    rng.manual_seed(42)
    dummy = torch.randint(0, tokenizer.vocab_size, (2, 32), generator=rng)

    with torch.no_grad():
        pt_out = model(dummy).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["logits"], {"input_ids": dummy.numpy()})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f"Max absolute difference: {max_diff:.2e}")

    if max_diff < atol:
        print("PASS")
        return True

    print(f"FAIL  (tolerance {atol:.0e})")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="mamba.onnx")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/mamba.yaml")
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    ok = verify(args.onnx, args.checkpoint, args.config, args.atol)
    sys.exit(0 if ok else 1)
