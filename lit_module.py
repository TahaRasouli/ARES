from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model import EssayScorer


def build_loss(loss_name: str) -> nn.Module:
    name = loss_name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f"Unknown loss: {loss_name}. Use one of: mse, mae, huber")


class EssayLightning(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        optimizer_name: str = "adamw",
        loss_name: str = "mse",
        warmup_ratio: float = 0.06,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = EssayScorer(model_name=model_name)
        self.crit = build_loss(loss_name)

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name.lower()
        self.warmup_ratio = warmup_ratio

    def _masked_multi_loss(self, preds: Dict[str, torch.Tensor], labels_list: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        # labels_list: length B, each is dict of available targets for that sample
        losses = []
        for i, lab in enumerate(labels_list):
            for k, y in lab.items():
                losses.append(self.crit(preds[k][i], y.to(preds[k].device)))
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(losses).mean()

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        preds = self.net(batch["input_ids"], batch["attention_mask"])
        loss = self._masked_multi_loss(preds, batch["labels_list"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        preds = self.net(batch["input_ids"], batch["attention_mask"])
        loss = self._masked_multi_loss(preds, batch["labels_list"])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt_name = self.optimizer_name
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif opt_name == "adafactor":
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                self.parameters(),
                lr=self.lr,
                relative_step=False,
                warmup_init=False,
                scale_parameter=False,
                weight_decay=self.weight_decay,
            )
        elif opt_name == "lion":
            # Lion is not in torch by default; fallback to AdamW if not installed
            try:
                from lion_pytorch import Lion  # pip install lion-pytorch
                optimizer = Lion(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            except Exception:
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("optimizer_name must be one of: adamw, adafactor, lion")

        # Scheduler: warmup + linear decay (recommended for transformers)
        if self.trainer is None or self.trainer.estimated_stepping_batches is None:
            return optimizer

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * self.warmup_ratio)

        from transformers.optimization import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
