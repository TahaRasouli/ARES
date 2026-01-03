from typing import Dict, List
import torch
import torch.nn as nn
import pytorch_lightning as pl

from eclipse_model import EclipseScorer, ECLIPSE_KEYS


class EclipseLightning(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = EclipseScorer(model_name)
        self.crit = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def _loss(self, preds: Dict[str, torch.Tensor], labels_list: List[Dict[str, torch.Tensor]]):
        losses = []
        for i, lab in enumerate(labels_list):
            for k in ECLIPSE_KEYS:
                losses.append(self.crit(preds[k][i], lab[k].to(self.device)))
        return torch.stack(losses).mean()

    def training_step(self, batch, batch_idx):
        preds = self.net(batch["input_ids"], batch["attention_mask"])
        loss = self._loss(preds, batch["labels_list"])

        self.train_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.net(batch["input_ids"], batch["attention_mask"])
        loss = self._loss(preds, batch["labels_list"])

        self.val_losses.append(loss.detach())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            avg = torch.stack(self.train_losses).mean().item()
            self.print(f"\nEpoch {self.current_epoch} | Train MSE: {avg:.4f}")
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            avg = torch.stack(self.val_losses).mean().item()
            self.print(f"Epoch {self.current_epoch} | Val   MSE: {avg:.4f}")
        self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
