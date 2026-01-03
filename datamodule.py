from typing import Optional, List, Dict, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset_unified import UnifiedEssayDataset, collate_fn
from data_io import load_json, load_jsonl, load_eclipse_csv


class EssayDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task1_jsonl_path: str,
        task2_json_path: str,
        eclipse_train_csv: str,
        eclipse_val_csv: str,
        model_name: str = "microsoft/deberta-v3-base",
        batch_size: int = 4,
        max_length: int = 512,
        num_workers: int = 4,
    ):
        super().__init__()
        self.task1_jsonl_path = task1_jsonl_path
        self.task2_json_path = task2_json_path
        self.eclipse_train_csv = eclipse_train_csv
        self.eclipse_val_csv = eclipse_val_csv

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

        self.tokenizer = None
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        task1 = load_jsonl(self.task1_jsonl_path)
        task2 = load_json(self.task2_json_path)
        eclipse_train = load_eclipse_csv(self.eclipse_train_csv)
        eclipse_val = load_eclipse_csv(self.eclipse_val_csv)

        # Ensure required fields exist (prompt/essay). Your existing task2 JSON might use Topic/essay
        def normalize_record(r: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(r)
            if "prompt" not in out and "Topic" in out:
                out["prompt"] = out["Topic"]
            if "essay" not in out and "content" in out:
                out["essay"] = out["content"]
            return out

        task1 = [normalize_record(r) for r in task1]
        task2 = [normalize_record(r) for r in task2]

        train_records = task1 + task2 + eclipse_train
        val_records = eclipse_val  # keep validation clean/consistent

        self.train_ds = UnifiedEssayDataset(train_records, self.tokenizer, self.max_length)
        self.val_ds = UnifiedEssayDataset(val_records, self.tokenizer, self.max_length)

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
