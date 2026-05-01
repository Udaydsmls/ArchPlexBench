import torch.nn as nn


class LSTMLM(nn.Module):
    """Multi-layer LSTM language model."""

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
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.drop(self.embed(x))
        out, _ = self.lstm(emb)
        return self.head(self.drop(out))
