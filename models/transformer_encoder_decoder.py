import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional causal mask."""

    def __init__(self, d_model, n_heads, dropout, causal):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.attn_dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        dp = self.attn_dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=dp)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from decoder, keys and values from encoder."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        B, T, C = x.shape
        S = enc_out.size(1)
        H, dh = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)
        k, v = self.kv_proj(enc_out).split(C, dim=-1)
        k = k.view(B, S, H, dh).transpose(1, 2)
        v = v.view(B, S, H, dh).transpose(1, 2)
        dp = self.attn_dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=dp)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FFN(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Bidirectional self-attention encoder block."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, causal=False)
        self.ff = FFN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """Causal self-attention + cross-attention to encoder + FFN."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout, causal=True)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.ff = FFN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), enc_out)
        x = x + self.ff(self.ln3(x))
        return x


class EncoderDecoderLM(nn.Module):
    """
    Encoder-decoder transformer for language modeling.
    Encoder processes the full input sequence bidirectionally;
    decoder attends to encoder output causally to predict next tokens.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.enc_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dec_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        emb = self.drop(self.tok_emb(x) + self.pos_emb(pos))

        enc = emb
        for block in self.encoder:
            enc = block(enc)
        enc = self.enc_norm(enc)

        dec = emb
        for block in self.decoder:
            dec = block(dec, enc)
        dec = self.dec_norm(dec)

        return self.head(dec)
