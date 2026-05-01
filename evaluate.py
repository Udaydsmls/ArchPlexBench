import math
import torch
import torch.nn as nn


def compute_perplexity(model, loader, device):
    """Evaluate token-level perplexity on the given DataLoader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
            total_tokens += y.numel()

    return math.exp(total_loss / total_tokens)
