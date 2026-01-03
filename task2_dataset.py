from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from text_utils import normalize_paragraphs

TASK2_KEYS = ["TA", "CC", "LR", "GA"]


class Task2Dataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]

        prompt = r.get("prompt", "")
        if not prompt and "Topic" in r:
            prompt = r["Topic"]
        prompt = normalize_paragraphs(str(prompt))

        essay = normalize_paragraphs(str(r.get("essay", "")))

        text = f"PROMPT:\n{prompt}\n\nESSAY:\n{essay}"

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels: Dict[str, torch.Tensor] = {}
        for k in TASK2_KEYS:
            v = r.get(k, None)
            if v is None:
                continue
            labels[k] = torch.tensor(float(v), dtype=torch.float32)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "labels_list": [b["labels"] for b in batch],
    }
