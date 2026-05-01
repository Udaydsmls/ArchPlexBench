import torch.nn as nn


class LSTMLM(nn.Module):
    """Multi-layer LSTM language model with weight-tied input/output embeddings."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        # Project back to embed_dim before the tied head when dims differ
        self.out_proj = (
            nn.Linear(hidden_dim, embed_dim, bias=False)
            if hidden_dim != embed_dim
            else nn.Identity()
        )
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.embed.weight = self.head.weight

    def forward(self, x):
        emb = self.drop(self.embed(x))
        out, _ = self.lstm(emb)
        return self.head(self.drop(self.out_proj(out)))
