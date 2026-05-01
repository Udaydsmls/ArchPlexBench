import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast


class WikiTextDataset(Dataset):
    """WikiText-2 tokenized into fixed-length chunks for language modeling."""

    def __init__(self, split, seq_len, tokenizer):
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n".join(raw["text"])
        tokens = tokenizer.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def get_tokenizer():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def get_dataloaders(seq_len, batch_size, tokenizer, num_workers=2):
    """Return train, validation, and test DataLoaders for WikiText-2."""
    loaders = {}
    for split in ("train", "validation", "test"):
        ds = WikiTextDataset(split, seq_len, tokenizer)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            pin_memory=True,
            num_workers=num_workers,
        )
    return loaders["train"], loaders["validation"], loaders["test"]
