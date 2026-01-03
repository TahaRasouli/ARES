from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from text_utils import normalize_paragraphs

IELTS_KEYS = ["TA", "TR", "CC", "LR", "GA"]


class IELTSStageDataset(Dataset):
    """
    Mixed Task1 + Task2 dataset:
      - Task 1 uses: TR, CC, LR, GA
      - Task 2 uses: TA, CC, LR, GA
    Missing labels are simply absent (masked loss).
    """
    def __init__(self, records: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
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

        labels = {}
        for k in IELTS_KEYS:
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
