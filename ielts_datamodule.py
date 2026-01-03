import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ielts_dataset import IELTSStageDataset, collate_fn


def load_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of records")
    return data


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    records = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_record(r: Dict[str, Any], task: str) -> Dict[str, Any]:
    """
    task: "task1" or "task2"
    We unify schema to: prompt, essay, TA/TR/CC/LR/GA
    """
    out = dict(r)

    # prompt field
    if "prompt" not in out:
        if "Topic" in out:
            out["prompt"] = out["Topic"]
        elif "topic" in out:
            out["prompt"] = out["topic"]
        elif "subject" in out:
            out["prompt"] = out["subject"]
        else:
            out["prompt"] = ""

    # essay field
    if "essay" not in out:
        if "content" in out:
            out["essay"] = out["content"]
        else:
            out["essay"] = ""

    # ensure correct key exists per task
    # Task1 -> TR, Task2 -> TA (may still have missing/nulls)
    if task == "task1":
        if "TR" not in out:
            # common HF fields (already converted sometimes)
            if "task_response_score" in out and out["task_response_score"] is not None:
                out["TR"] = float(out["task_response_score"])
    elif task == "task2":
        if "TA" not in out:
            # nothing to infer safely; keep missing
            pass

    return out


def split_train_val(records: List[Dict[str, Any]], val_ratio: float = 0.15, seed: int = 42):
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    n_val = max(1, int(len(records) * val_ratio))
    val_set = set(idxs[:n_val])
    train = [records[i] for i in range(len(records)) if i not in val_set]
    val = [records[i] for i in range(len(records)) if i in val_set]
    return train, val


class IELTSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task1_jsonl_path: str,
        task2_json_path: str,
        model_name: str = "microsoft/deberta-v3-base",
        batch_size: int = 4,
        max_length: int = 512,
        num_workers: int = 4,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.task1_jsonl_path = task1_jsonl_path
        self.task2_json_path = task2_json_path
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

        task1 = load_jsonl(self.task1_jsonl_path)
        task2 = load_json(self.task2_json_path)

        task1 = [normalize_record(r, "task1") for r in task1]
        task2 = [normalize_record(r, "task2") for r in task2]

        all_records = task1 + task2
        train_records, val_records = split_train_val(all_records, self.val_ratio, self.seed)

        self.train_ds = IELTSStageDataset(train_records, self.tokenizer, self.max_length)
        self.val_ds = IELTSStageDataset(val_records, self.tokenizer, self.max_length)

        # optional debug (uncomment once if needed)
        # print("IELTS TRAIN:", len(self.train_ds), "IELTS VAL:", len(self.val_ds))

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
