import torch
import torch.nn as nn
import pytorch_lightning as pl
from model import EssayEncoder

class EclipseLightningModule(pl.LightningModule):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = EssayEncoder()
        self.head = nn.Linear(512, 7)
        self.criterion = nn.MSELoss()

        # store losses explicitly
        self.train_losses = []
        self.val_losses = []

    def forward(self, input_ids, attention_mask):
        repr = self.encoder(input_ids, attention_mask)
        return self.head(repr)

    def training_step(self, batch, batch_idx):
        preds = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(preds, batch["labels"])

        self.train_losses.append(loss.detach())
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(preds, batch["labels"])

        self.val_losses.append(loss.detach())
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            avg_train = torch.stack(self.train_losses).mean().item()
            print(f"\nEpoch {self.current_epoch} | Train MSE: {avg_train:.4f}")
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        if len(self.val_losses) > 0 and self.trainer.is_global_zero:
            avg_val = torch.stack(self.val_losses).mean().item()
            print(f"Epoch {self.current_epoch} | Val   MSE: {avg_val:.4f}")
        self.val_losses.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
