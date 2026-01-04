from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from text_utils import normalize_paragraphs
from task2_data import TASK2_KEYS


class Task2Dataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def _build_text(self, prompt: str, essay: str) -> str:
        prompt = normalize_paragraphs(prompt)
        essay = normalize_paragraphs(essay)
        return f"PROMPT:\n{prompt}\n\nESSAY:\n{essay}"

    def __getitem__(self, idx):
        r = self.records[idx]
        text = self._build_text(r.get("prompt", ""), r.get("essay", ""))

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # normalize labels to [0,1]
        labels = {k: torch.tensor(float(r[k]) / 9.0, dtype=torch.float32) for k in TASK2_KEYS if r.get(k) is not None}

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels_list = [b["labels"] for b in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels_list": labels_list}
