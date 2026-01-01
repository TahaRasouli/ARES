import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


RUBRICS = [
    "Overall", "Cohesion", "Syntax", "Vocabulary",
    "Phraseology", "Grammar", "Conventions"
]

class EclipseDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.prompts = df["prompt"].fillna("").tolist()
        self.essays = df["full_text"].fillna("").tolist()
        self.labels = df[RUBRICS].values.astype("float32")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        combined_text = f"PROMPT: {self.prompts[idx]} ESSAY: {self.essays[idx]}"
        enc = self.tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

class EclipseDataModule(pl.LightningDataModule):
    def __init__(self, train_csv_path, val_csv_path, tokenizer, batch_size=4, max_len=512):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len

    def setup(self, stage=None):
        train_df = pd.read_csv(self.train_csv_path)
        val_df = pd.read_csv(self.val_csv_path)

        self.train_ds = EclipseDataset(train_df, self.tokenizer, self.max_len)
        self.val_ds = EclipseDataset(val_df, self.tokenizer, self.max_len)

        # Helpful: print sizes once on rank 0
        if self.trainer is None or self.trainer.is_global_zero:
            print(f"Train size: {len(self.train_ds)} | Val size: {len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
