import torch
import torch.nn as nn
import pytorch_lightning as pl
from model import EssayEncoder

class EclipseLightningModule(pl.LightningModule):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = EssayEncoder()

        # 7 analytic rubric outputs
        self.head = nn.Linear(512, 7)

        self.criterion = nn.MSELoss()

    def forward(self, input_ids, attention_mask):
        repr = self.encoder(input_ids, attention_mask)
        return self.head(repr)

    def training_step(self, batch, batch_idx):
        preds = self(
            batch["input_ids"],
            batch["attention_mask"]
        )
        loss = self.criterion(preds, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(
            batch["input_ids"],
            batch["attention_mask"]
        )
        loss = self.criterion(preds, batch["labels"])
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr
        )

