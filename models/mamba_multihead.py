import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


@torch.compile
def _ssm_multihead(x_conv, A, dt, B_heads, C_heads, head_mix, D_param):
    """Fused multi-head SSM kernel: discretization, parallel prefix scan, and
    head mixing in one compiled region so TorchInductor co-fuses the entire
    5D elementwise chain instead of stopping at the scan boundary.
    """
    dt_e = dt.unsqueeze(-1).unsqueeze(-1)                       # (B, T, D, 1, 1)
    dA = torch.exp(A * dt_e)                                    # (B, T, D, P, N)
    dtx_e = (dt * x_conv).unsqueeze(-1).unsqueeze(-1)
    X = dtx_e * B_heads.unsqueeze(2)                            # (B, T, D, P, N)

    T = dA.shape[1]
    log_T = math.ceil(math.log2(T)) if T > 1 else 0
    pad_inner = (0, 0, 0, 0, 0, 0)
    A_run = dA
    X_run = X
    for d in range(log_T):
        step = 2 ** d
        if step >= T:
            break
        pad_spec = pad_inner + (step, 0)
        A_prev = F.pad(A_run[:, :-step], pad_spec, value=1.0)
        X_prev = F.pad(X_run[:, :-step], pad_spec, value=0.0)
        X_run = A_run * X_prev + X_run
        A_run = A_run * A_prev

    y_heads = (X_run * C_heads.unsqueeze(2)).sum(-1)            # (B, T, D, P)
    y = (y_heads * head_mix).sum(-1) + x_conv * D_param         # (B, T, D)
    return y


class SelectiveSSM(nn.Module):
    """
    Multi-head selective state space model.
    Each of the n_heads heads maintains its own B and C projections and a
    separate slice of the A matrix, allowing the model to attend to different
    frequency components of the input. Outputs are mixed via a learnable
    head_mix weight before the output projection.
    """

    def __init__(self, d_model, d_state, d_conv, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).unsqueeze(0).expand(d_model, n_heads, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        self.B_proj = nn.Linear(d_model, n_heads * d_state, bias=False)
        self.C_proj = nn.Linear(d_model, n_heads * d_state, bias=False)
        self.head_mix = nn.Parameter(torch.full((d_model, n_heads), 1.0 / n_heads))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        P, N = self.n_heads, self.d_state

        x_ssm, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        A = (-torch.exp(self.A_log))[None, None]                 # (1, 1, D, P, N)
        B_heads = self.B_proj(x_conv).view(B, T, P, N)
        C_heads = self.C_proj(x_conv).view(B, T, P, N)

        y = _ssm_multihead(x_conv, A, dt, B_heads, C_heads, self.head_mix, self.D)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaMultiHeadBlock(nn.Module):
    """Multi-head Mamba residual block with pre-norm."""

    def __init__(self, d_model, d_state, d_conv, n_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, n_heads)
        self.drop = nn.Dropout(dropout)

    def _forward(self, x):
        return x + self.drop(self.ssm(self.norm(x)))

    def forward(self, x):
        if self.training:
            return ckpt.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class MambaMultiHeadLM(nn.Module):
    """Multi-head Mamba language model."""

    def __init__(self, vocab_size, d_model, n_layers, d_state, d_conv, n_heads, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [MambaMultiHeadBlock(d_model, d_state, d_conv, n_heads, dropout)
             for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed.weight = self.head.weight

    def forward(self, x):
        x = self.drop(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_f(x))
