from models.transformer import TransformerLM
from models.lstm import LSTMLM
from models.mamba import MambaLM


def build_model(cfg, vocab_size):
    """Instantiate a model from a config dict."""
    kind = cfg["model"]
    if kind == "transformer":
        return TransformerLM(
            vocab_size=vocab_size,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            max_seq_len=cfg["max_seq_len"],
        )
    if kind == "lstm":
        return LSTMLM(
            vocab_size=vocab_size,
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
        )
    if kind == "mamba":
        return MambaLM(
            vocab_size=vocab_size,
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            d_state=cfg["d_state"],
            d_conv=cfg["d_conv"],
            expand=cfg["expand"],
            dropout=cfg["dropout"],
        )
    raise ValueError(f"Unknown model type: {kind}")
