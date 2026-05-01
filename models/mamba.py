import torch.nn as nn

_mamba_available = False
try:
    from mamba_ssm import Mamba as _MambaOp
    _mamba_available = True
except ImportError:
    pass


class MambaBlock(nn.Module):
    """Mamba SSM block with pre-norm and residual connection."""

    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        if not _mamba_available:
            raise ImportError("Install mamba-ssm: pip install mamba-ssm causal-conv1d")
        self.mixer = _MambaOp(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class MambaLM(nn.Module):
    """Mamba state-space model language model."""

    def __init__(self, vocab_size, d_model, n_layers, d_state, d_conv, expand, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed.weight = self.head.weight  # weight tying

    def forward(self, x):
        x = self.drop(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_f(x))
