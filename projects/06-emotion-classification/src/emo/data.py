import numpy as np
import torch

from emo.labels import multihot

ge_id = "google-research-datasets/go_emotions"
dair_id = "dair-ai/emotion"


def load_ge():
    # returns {split: (texts, multi-hot labels)} for train / validation / test
    from datasets import load_dataset

    ds = load_dataset(ge_id, "simplified")
    return {s: (list(ds[s]["text"]), multihot(ds[s]["labels"])) for s in ["train", "validation", "test"]}


def load_dair(split="test"):
    from datasets import load_dataset

    ds = load_dataset(dair_id, "split")[split]
    return list(ds["text"]), np.array(ds["label"])


class TextDs(torch.utils.data.Dataset):
    # pre-tokenizes once, padding happens per batch in collate
    def __init__(self, x, y, tok, max_len=128, soft=None):
        self.enc = tok(x, truncation=True, max_length=max_len)["input_ids"]
        self.y = y
        self.soft = soft

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        d = {"ids": self.enc[i], "y": self.y[i]}
        if self.soft is not None:
            d["soft"] = self.soft[i]
        return d


def collate(batch, pad_id=1):
    n = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = torch.tensor(b["ids"])
        mask[i, : len(b["ids"])] = 1
    out = {"input_ids": ids, "attention_mask": mask, "y": torch.tensor(np.stack([b["y"] for b in batch]))}
    if "soft" in batch[0]:
        out["soft"] = torch.tensor(np.stack([b["soft"] for b in batch]))
    return out
