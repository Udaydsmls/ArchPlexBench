# ArchPlexBench

A from-scratch benchmark that trains six sequence modelling architectures under identical conditions and compares them on perplexity. Every model is implemented in pure PyTorch — no high-level wrappers — so the architectural differences are transparent and the playing field is level.

---

## Architectures

| Model | File | Key idea |
|---|---|---|
| Transformer (decoder-only) | `models/transformer_decoder.py` | Causal self-attention with Flash Attention (`F.scaled_dot_product_attention`) |
| Transformer (encoder-decoder) | `models/transformer_encoder_decoder.py` | Bidirectional encoder + causal decoder with cross-attention over the full input sequence |
| LSTM | `models/lstm.py` | Multi-layer LSTM with weight-tied embeddings and optional hidden→embed projection |
| xLSTM | `models/xlstm.py` | Beck et al., NeurIPS 2024. Interleaved mLSTM (matrix memory, outer-product writes) and sLSTM (exponential gating, cross-head memory mixing via shared recurrent weight) |
| Mamba | `models/mamba.py` | Selective SSM built from scratch: ZOH-discretised diagonal A, sequential recurrence, input-dependent B/C/Δ |
| Mamba (multi-head) | `models/mamba_multihead.py` | Multi-head extension of Mamba: per-head B/C projections with a learnable `head_mix` weight |

All models use weight-tied input/output embeddings and are tuned to sit in the **13–18 M parameter** range for a fair comparison.

---

## Dataset

**WikiText-2** via HuggingFace `datasets`, tokenised with the GPT-2 BPE tokeniser (vocab size 50 257). Sequences are split into fixed-length non-overlapping chunks; perplexity is reported at the token level on the held-out test split.

---

## Results

Results are written to `results/benchmark.csv` and `results/comparison.png` after a run.

| model | val_ppl | test_ppl | params_M |
|---|---|---|---|
| *(run the benchmark to populate)* | | | |

---

## Quickstart

### Google Colab (recommended)

Open `run_colab.ipynb`, set the runtime to **T4 GPU**, and run all cells. The notebook installs dependencies, clones the repo, runs the full benchmark, and displays the results table and bar chart.

### Local

```bash
# 1. create / activate your environment
conda activate ML

# 2. install dependencies
pip install -r requirements.txt

# 3. run the full benchmark (all 6 models)
python benchmark.py --epochs 20 --batch_size 32 --seq_len 256

# 4. train a single model
python train.py --config configs/transformer_decoder.yaml --epochs 20
```

---

## Project structure

```
ArchPlexBench/
├── models/
│   ├── transformer_decoder.py        # decoder-only transformer
│   ├── transformer_encoder_decoder.py# encoder-decoder transformer
│   ├── lstm.py                       # LSTM baseline
│   ├── xlstm.py                      # xLSTM (sLSTM + mLSTM)
│   ├── mamba.py                      # Mamba SSM
│   ├── mamba_multihead.py            # multi-head Mamba
│   └── __init__.py                   # build_model factory
├── data/
│   └── dataset.py                    # WikiText-2 loading and tokenisation
├── configs/
│   ├── transformer_decoder.yaml
│   ├── encoder_decoder.yaml
│   ├── lstm.yaml
│   ├── xlstm.yaml
│   ├── mamba.yaml
│   └── mamba_multihead.yaml
├── train.py                          # shared training loop
├── evaluate.py                       # perplexity computation
├── benchmark.py                      # orchestrates all models, plots results
├── run_colab.ipynb                   # one-click Colab notebook
└── requirements.txt
```

---

## Configuration

Each model has a YAML config in `configs/`. Hyperparameters can be changed there without touching the model code. To add a new architecture, implement the `forward(x) -> logits` interface, register it in `models/__init__.py`, and add a config file.

```yaml
# configs/transformer_decoder.yaml
model: transformer_decoder
d_model: 256
n_heads: 8
n_layers: 4
d_ff: 1024
dropout: 0.1
max_seq_len: 256
```

---

## Requirements

```
torch >= 2.0
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
