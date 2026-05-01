import torch
import torch.nn as nn
import torch.nn.functional as F


class sLSTMCell(nn.Module):
    """
    Scalar LSTM cell with exponential gating and stabilization.
    Cross-head memory mixing is achieved via a shared recurrent projection R
    that maps the full concatenated hidden state (all heads) to per-head gates.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        dh = self.d_head
        gate_dim = n_heads * (2 + 2 * dh)  # log_i, log_f, z (dh), o (dh) per head
        self.W = nn.Linear(d_model, gate_dim, bias=True)
        self.R = nn.Linear(d_model, gate_dim, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head

        c = x.new_zeros(B, H, dh)
        n = x.new_zeros(B, H, 1)
        h_prev = x.new_zeros(B, D)
        m = x.new_full((B, H), -1e9)

        ys = []
        for t in range(L):
            g = (self.W(x[:, t]) + self.R(h_prev)).view(B, H, 2 + 2 * dh)
            log_i = g[:, :, 0]
            log_f = g[:, :, 1]
            z = torch.tanh(g[:, :, 2:2 + dh])
            o = torch.sigmoid(g[:, :, 2 + dh:])

            m_new = torch.maximum(log_f + m, log_i)
            f = torch.exp(log_f + m - m_new).unsqueeze(-1)   # (B, H, 1)
            i = torch.exp(log_i - m_new).unsqueeze(-1)        # (B, H, 1)
            m = m_new

            c = f * c + i * z
            n = f * n + i
            h_t = o * torch.tanh(c / n.abs().clamp(min=1))
            h_prev = h_t.reshape(B, D)
            ys.append(h_prev)

        return torch.stack(ys, dim=1)  # (B, L, D)


class mLSTMCell(nn.Module):
    """
    Matrix LSTM cell: replaces the scalar cell state with a per-head matrix memory
    C ∈ R^{d_head × d_head} updated via outer-product writes (v ⊗ k).
    Gates are input-only (no recurrent connections), making it parallelisable in theory.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        dh = self.d_head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.i_gate = nn.Linear(d_model, n_heads, bias=True)   # scalar per head
        self.f_gate = nn.Linear(d_model, n_heads, bias=True)
        self.o_gate = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x):
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, L, H, dh)
        k = self.k_proj(x).view(B, L, H, dh) / (dh ** 0.5)
        v = self.v_proj(x).view(B, L, H, dh)
        log_i = self.i_gate(x)   # (B, L, H)
        log_f = self.f_gate(x)
        o = torch.sigmoid(self.o_gate(x))  # (B, L, D)

        C = x.new_zeros(B, H, dh, dh)
        n = x.new_zeros(B, H, dh)
        m = x.new_full((B, H), -1e9)

        ys = []
        for t in range(L):
            m_new = torch.maximum(log_f[:, t] + m, log_i[:, t])
            f = torch.exp(log_f[:, t] + m - m_new)  # (B, H)
            i = torch.exp(log_i[:, t] - m_new)
            m = m_new

            vk = v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)       # (B, H, dh, dh)
            C = f[:, :, None, None] * C + i[:, :, None, None] * vk
            n = f[:, :, None] * n + i[:, :, None] * k[:, t]

            Cq = torch.einsum("bhij,bhj->bhi", C, q[:, t])           # (B, H, dh)
            denom = (n * q[:, t]).sum(-1).abs().clamp(min=1)          # (B, H)
            ys.append((Cq / denom[:, :, None]).reshape(B, D))

        return torch.stack(ys, dim=1) * o  # (B, L, D)


class sLSTMBlock(nn.Module):
    """sLSTM residual block with pre-norm and post-norm."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cell = sLSTMCell(d_model, n_heads)
        self.out_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.out_norm(self.cell(self.norm(x))))


class mLSTMBlock(nn.Module):
    """mLSTM residual block with pre-norm and post-norm."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cell = mLSTMCell(d_model, n_heads)
        self.out_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.out_norm(self.cell(self.norm(x))))


class xLSTMLM(nn.Module):
    """
    xLSTM language model (Beck et al., NeurIPS 2024).
    Stacks mLSTM blocks (matrix memory) and sLSTM blocks (scalar memory with
    exponential gating). slstm_at controls which layer indices use sLSTM;
    all others use mLSTM. The paper recommends a heavy mLSTM majority.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, dropout, slstm_at):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        blocks = []
        for i in range(n_layers):
            if i in slstm_at:
                blocks.append(sLSTMBlock(d_model, n_heads, dropout))
            else:
                blocks.append(mLSTMBlock(d_model, n_heads, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed.weight = self.head.weight

    def forward(self, x):
        x = self.drop(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_f(x))
