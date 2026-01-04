import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from task2_dataset import Task2Dataset, collate_fn


def load_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Task2 JSON must contain a list of objects")
    return data


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_task2(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".json":
        return load_json(path)
    if p.suffix == ".jsonl":
        return load_jsonl(path)
    raise ValueError("Task2 file must be .json or .jsonl")


def split_train_val(records: List[Dict[str, Any]], val_ratio: float = 0.15, seed: int = 42):
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    n_val = max(1, int(len(records) * val_ratio))
    val_set = set(idxs[:n_val])

    train = [records[i] for i in range(len(records)) if i not in val_set]
    val = [records[i] for i in range(len(records)) if i in val_set]
    return train, val

def _clean_band(x):
    if x is None:
        return None
    v = float(x)
    if v < 0 or v > 9:
        return None
    return v

def has_all_four(r):
    return all(r.get(k) is not None for k in ["TA", "CC", "LR", "GA"])

class Task2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        task2_path: str,
        model_name: str = "microsoft/deberta-v3-base",
        batch_size: int = 4,
        max_length: int = 512,
        num_workers: int = 4,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.task2_path = task2_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed

        self.tokenizer = None
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        records = load_task2(self.task2_path)

        # Normalize schema for safety
        norm = []
        for r in records:
            rr = dict(r)
            if "prompt" not in rr and "Topic" in rr:
                rr["prompt"] = rr["Topic"]
            if "essay" not in rr and "content" in rr:
                rr["essay"] = rr["content"]
            norm.append(rr)

        train_records, val_records = split_train_val(norm, self.val_ratio, self.seed)
        self.train_ds = Task2Dataset(train_records, self.tokenizer, self.max_length)
        self.val_ds = Task2Dataset(val_records, self.tokenizer, self.max_length)
        records = [r for r in records if has_all_four(r)]


        # Optional debug
        print("Task2 Train:", len(self.train_ds), "Task2 Val:", len(self.val_ds))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
