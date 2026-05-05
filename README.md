# ArchPlexBench

A from-scratch benchmark that trains five sequence-modelling architectures under identical conditions on WikiText-2 and compares them on token-level perplexity. Every model is implemented in pure PyTorch — no high-level wrappers, no `mamba-ssm` CUDA kernel — so the architectural differences are transparent and the playing field is level.

---

## Architectures

| Model | File | Key idea |
|---|---|---|
| Transformer (decoder-only) | `models/transformer_decoder.py` | Causal self-attention with Flash Attention (`F.scaled_dot_product_attention`) |
| Transformer (encoder-decoder) | `models/transformer_encoder_decoder.py` | Bidirectional encoder + causal decoder with cross-attention over the full input sequence |
| LSTM | `models/lstm.py` | Multi-layer LSTM with weight-tied embeddings and a hidden→embed projection |
| Mamba | `models/mamba.py` | Selective SSM from scratch: ZOH-discretised diagonal A, input-dependent B/C/Δ, parallel prefix scan |
| Mamba (multi-head) | `models/mamba_multihead.py` | Multi-head extension of Mamba: per-head B/C projections with a learnable `head_mix` weight |

> An xLSTM implementation lives in `models/xlstm.py` for reference, but it is excluded from the benchmark — its mLSTM matrix-memory updates ran an order of magnitude slower than every other architecture in pure PyTorch and would have dominated runtime without telling us anything new about the recurrence.

All models are tuned to land in the **5.7–6.1 M parameter range**. The constraint is dominated by the GPT-2 BPE embedding (50 257 × `d_model`), which forces `d_model = 96` once you also want enough budget for layers — pushing to `d_model = 128` would cost 6.4 M just on embeddings. Weights are tied between input embedding and output head everywhere.

---

## Dataset

**WikiText-2** via HuggingFace `datasets`, tokenised with the GPT-2 BPE tokeniser (vocab size 50 257). Sequences are split into fixed-length non-overlapping chunks of 256 tokens; perplexity is reported at the token level on the held-out test split.

---

## Results

20 epochs, batch size 32, sequence length 256, AdamW (lr=3e-4, weight_decay=0.1) with cosine annealing, gradient clip 1.0. Identical optimiser settings across all five models. Run on a single Colab T4.

| model | val_ppl | test_ppl | params_M | per-epoch | total (20ep) |
|---|---:|---:|---:|---:|---:|
| **mamba** | **207.29** | **214.61** | 5.96 | ~458 s | 2 h 33 min |
| mamba_multihead | 227.95 | 233.40 | 5.76 | ~593 s | 3 h 18 min |
| lstm | 497.22 | 506.28 | 5.74 | ~58 s | 19.3 min |
| transformer_decoder | 645.90 | 652.90 | 5.93 | ~65 s | 21.6 min |
| encoder_decoder | 686.02 | 689.49 | 6.08 | ~71 s | 23.9 min |

End-to-end benchmark wall-clock: **~6 h 56 min**. Plotted in [`results/comparison.png`](results/comparison.png).

---

## Analysis

### Perplexity

**Mamba dominates.** It reaches a test PPL of 214.6, roughly **3× lower than the decoder-only Transformer** at matched parameter count. The multi-head Mamba lands close behind at 233.4 — the per-head B/C projections did not buy more capacity than they cost on this data scale. Both Mamba models are essentially converged by epoch 20 (final-epoch PPL change < 0.1).

**LSTM beats both Transformers.** This is the headline finding the small-data regime has reproduced for years: at 5.7 M parameters and 2 M training tokens, the inductive bias of a recurrence outweighs the flexibility of attention. The LSTM hidden state is also wider (`hidden_dim=256`) than the Transformer's `d_model=96`, so its per-step computation has more room.

**The Transformers are under-trained**, not broken. Both are still descending by ~3 PPL per epoch at epoch 20 — a 50-epoch run would close some of the gap. They suffer most from the `d_model=96` constraint: at this width the attention layer has only 4 heads of `head_dim=24`, and there is no per-model LR warmup to compensate. Per-model tuning would help them; we deliberately avoided it for fairness.

**Encoder-decoder is the worst performer.** The encoder runs bidirectionally over the full sequence, so it sees the same context as a plain LM but spends half the layers (4 each instead of 8) on bidirectional encoding rather than causal prediction. It's the wrong shape for autoregressive language modelling and the numbers reflect that.

### Speed

Mamba is **~8× slower per epoch than the Transformer**, and multi-head Mamba is **~10× slower**. This is the headline cost of the from-scratch implementation. The reasons are structural rather than fixable in pure PyTorch:

1. **No fused CUDA kernel.** The Mamba paper's published speed numbers come from `selective_scan_fn`, a hand-tuned kernel that fuses all 256 timesteps into a single launch and keeps state in shared memory. We deliberately did not use `mamba-ssm`.
2. **Sequential dependency in the SSM recurrence.** `h_t = a_t · h_{t-1} + b_t` cannot be parallelised the way attention can. We mitigate this with a Hillis-Steele **parallel prefix scan** (8 sequential GPU iterations instead of 256), but each iteration still moves the full `(B, T, D, N)` activation through HBM.
3. **Gradient checkpointing on every Mamba block** is required to keep activation memory under T4's 16 GB. The recompute during backward adds ~50 % wall-clock per layer.
4. **Multi-head amplifies all of the above.** Its scan operates on 5D `(B, T, D, P, N)` tensors that are 2–3× heavier than the single-head's 4D ones, even after halving `d_state` from 16 to 8.

What we *did* apply, and the order of impact:

- `torch.compile` on the scan kernel — TorchInductor fuses the `pad → mul → mul → add` chain into one kernel per scan iteration, cutting kernel-launch overhead and HBM traffic.
- Combined `B_proj` / `C_proj` linears (multi-head) — replaces six small matmuls per layer with one batched matmul.
- Pre-computing `(δ · x)` before the 5D broadcast (multi-head) — saves one 150 MB intermediate per layer per forward.
- The whole multi-head SSM (discretisation + scan + head mix) sits inside a single `@torch.compile` region so fusion is not interrupted at the scan boundary.

Without these, multi-head Mamba would not finish 20 epochs on T4 in a tractable time. Even with them, the gap to attention is real.

### Cost / quality trade-off

| | LSTM | Transformer | Mamba |
|---|---|---|---|
| Wall time (20ep) | 19 min | 22 min | **153 min** |
| Test PPL | 506 | 653 | **215** |
| min · PPL (lower better) | 9 614 | 14 366 | 32 822 |

The LSTM is the **best PPL-per-second** in this benchmark — under 20 minutes of T4 time for a respectable 506 PPL. The Mamba quality win costs an order of magnitude more compute. If you are going to pay that compute cost and want production-grade speed, swap our pure-PyTorch SSM for `mamba-ssm`'s CUDA path; you'll get roughly the same perplexity in roughly Transformer wall-clock.

### Caveats

- 20 epochs is short. Transformers are clearly still learning and would close some of the gap with more compute. The Mamba numbers, in contrast, are near their training-data ceiling.
- Identical hyperparameters across architectures favours the model whose default behaviour matches them best. Per-architecture LR / warmup schedules would shuffle the rankings somewhat (most likely helping the Transformers).
- WikiText-2 is small — 2 M training tokens. Results here say nothing about how these architectures rank at the 1 B+ token scale.

---

## Quickstart

### Google Colab (recommended)

Open `run_colab.ipynb`, set the runtime to **T4 GPU**, and run all cells. The notebook installs dependencies, clones the repo, runs the full benchmark, and displays the results table and bar chart. Expect ~7 hours end-to-end.

### Local

```bash
conda activate ML
pip install -r requirements.txt

# full benchmark (all 5 models)
python benchmark.py --epochs 20 --batch_size 32 --seq_len 256

# single model
python train.py --config configs/mamba.yaml --epochs 20
```

For a quick sanity check, drop `--epochs 3`.

---

## Project structure

```
ArchPlexBench/
├── models/
│   ├── transformer_decoder.py        # decoder-only transformer
│   ├── transformer_encoder_decoder.py# encoder-decoder transformer
│   ├── lstm.py                       # LSTM baseline
│   ├── xlstm.py                      # xLSTM (reference, excluded from benchmark)
│   ├── mamba.py                      # Mamba SSM
│   ├── mamba_multihead.py            # multi-head Mamba
│   ├── scan.py                       # torch.compile'd parallel prefix scan
│   └── __init__.py                   # build_model factory
├── data/
│   └── dataset.py                    # WikiText-2 loading and tokenisation
├── configs/                          # one YAML per model, ~6 M params each
├── train.py                          # shared training loop
├── evaluate.py                       # perplexity computation
├── benchmark.py                      # orchestrates all models, plots results
├── results/
│   ├── benchmark.csv                 # final table
│   └── comparison.png                # bar charts (PPL, params)
├── run_colab.ipynb                   # one-click Colab notebook
└── requirements.txt
```

---

## Configuration

Each model has a YAML config in `configs/`. Hyperparameters can be changed there without touching the model code. To add a new architecture, implement `forward(x) -> logits`, register it in `models/__init__.py`, and add a config file.

```yaml
# configs/mamba.yaml
model: mamba
d_model: 96
n_layers: 16
d_state: 16
d_conv: 4
expand: 2
dropout: 0.1
```

---

## Requirements

```
torch >= 2.0      # 2.1+ recommended for torch.compile + checkpoint composition
transformers >= 4.35
datasets >= 2.14
pyyaml >= 6.0
pandas >= 2.0
matplotlib >= 3.7
```

---

## References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023
- Beck et al., *xLSTM: Extended Long Short-Term Memory*, NeurIPS 2024
