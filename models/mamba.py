import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from models.scan import parallel_scan


class SelectiveSSM(nn.Module):
    """Input-dependent state space model (S6) with ZOH-discretized diagonal A."""

    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_state = d_state
        dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -(dt_rank**-0.5), dt_rank**-0.5)

        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        self.dt_proj.bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        delta_pre, B_ssm, C = self.x_proj(x).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_pre))
        A = -torch.exp(self.A_log.float())

        dA = torch.exp(delta.unsqueeze(-1) * A)
        dBu = (delta * x).unsqueeze(-1) * B_ssm.unsqueeze(2)

        h = parallel_scan(dA, dBu)
        y = (h * C[:, :, None, :]).sum(-1)
        return y + self.D * x


class MambaBlock(nn.Module):
    """Mamba residual block: causal depthwise conv followed by selective SSM with SiLU gate."""

    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv, groups=d_inner,
            padding=d_conv - 1, bias=True,
        )
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _forward(self, x):
        residual = x
        xz = self.in_proj(self.norm(x))
        x_branch, z = xz.split(self.d_inner, dim=-1)

        L = x_branch.size(1)
        x_conv = self.conv1d(x_branch.transpose(1, 2))[..., :L]
        x_conv = F.silu(x_conv.transpose(1, 2))

        y = self.ssm(x_conv) * F.silu(z)
        return residual + self.out_proj(y)

    def forward(self, x):
        if self.training:
            return ckpt.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class MambaLM(nn.Module):
    """Mamba state-space language model."""

    def __init__(self, vocab_size, d_model, n_layers, d_state, d_conv, expand, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed.weight = self.head.weight

    def forward(self, x):
        x = self.drop(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_f(x))
