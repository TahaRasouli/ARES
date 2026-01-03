import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from eclipse_dataset import EclipseDataset, collate_fn
from data_io import load_eclipse_csv


class EclipseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        model_name: str,
        batch_size: int = 4,
        max_length: int = 512,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        train_records = load_eclipse_csv(self.train_csv)
        val_records = load_eclipse_csv(self.val_csv)

        self.train_ds = EclipseDataset(train_records, self.tokenizer, self.max_length)
        self.val_ds = EclipseDataset(val_records, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
