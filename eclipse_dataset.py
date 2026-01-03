from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from text_utils import normalize_paragraphs


ECLIPSE_KEYS = [
    "Overall",
    "Cohesion",
    "Syntax",
    "Vocabulary",
    "Phraseology",
    "Grammar",
    "Conventions",
]


class EclipseDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        prompt = normalize_paragraphs(r.get("prompt", ""))
        essay = normalize_paragraphs(r.get("essay", ""))

        text = f"PROMPT:\n{prompt}\n\nESSAY:\n{essay}"

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = {
            k: torch.tensor(float(r[k]), dtype=torch.float32)
            for k in ECLIPSE_KEYS
            if k in r and r[k] is not None
        }

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels_list": [b["labels"] for b in batch],
    }
