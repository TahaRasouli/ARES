from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from task2_data import load_task2, clean_and_filter_task2, split_train_val
from task2_dataset import Task2Dataset, collate_fn


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
        enforce_half_bands: bool = False,
        max_gap: float = 4.0,
    ):
        super().__init__()
        self.task2_path = task2_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        self.enforce_half_bands = enforce_half_bands
        self.max_gap = max_gap

        self.tokenizer = None
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        records = load_task2(self.task2_path)
        records = clean_and_filter_task2(
            records,
            require_all_four=True,
            enforce_half_bands=self.enforce_half_bands,
            max_gap=self.max_gap,
        )

        train_records, val_records = split_train_val(records, val_ratio=self.val_ratio, seed=self.seed)

        self.train_ds = Task2Dataset(train_records, self.tokenizer, self.max_length)
        self.val_ds = Task2Dataset(val_records, self.tokenizer, self.max_length)

        rank_zero_info(
            f"Task2 Train: {len(self.train_ds)} | Task2 Val: {len(self.val_ds)} | "
            f"max_length={self.max_length} enforce_half_bands={self.enforce_half_bands} max_gap={self.max_gap}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
