import pytorch_lightning as L
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from dataset import IELTSDataset

class IELTSDataModule(L.LightningDataModule):
    def __init__(self, train_path, model_name, batch_size=8):
        super().__init__()
        self.train_path = train_path
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage=None):
        full_dataset = IELTSDataset(self.train_path, self.tokenizer)
        train_size = int(0.9 * len(full_dataset))
        self.train_ds, self.val_ds = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)
