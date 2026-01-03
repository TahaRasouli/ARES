from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from text_utils import normalize_paragraphs


ALL_TARGET_KEYS = [
    # IELTS
    "TA", "TR", "CC", "LR", "GA",
    # ECLIPSE
    "Overall", "Cohesion", "Syntax", "Vocabulary",
    "Phraseology", "Grammar", "Conventions",
]


class UnifiedEssayDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def _build_text(self, prompt: str, essay: str) -> str:
        prompt = normalize_paragraphs(prompt)
        essay = normalize_paragraphs(essay)
        return f"PROMPT:\n{prompt}\n\nESSAY:\n{essay}"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        prompt = r.get("prompt", "")
        essay = r.get("essay", "")

        text = self._build_text(prompt, essay)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = {}
        for k in ALL_TARGET_KEYS:
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
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)

    # labels is a list of dicts with variable keys per sample
    labels_list = [b["labels"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels_list": labels_list,
    }
