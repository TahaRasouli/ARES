from typing import Dict, Any, List
import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers.optimization import get_linear_schedule_with_warmup
from ielts_model import IELTSScorer, IELTS_KEYS


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError("loss must be one of: mse, mae, huber")


def load_stage1a_weights_into_ielts(model: nn.Module, stage1a_ckpt_path: str):
    """
    Loads ONLY the shared backbone weights from Stage 1A into the IELTS model:
      - net.encoder.*
      - net.pool.*
      - net.proj.*
      - net.act.* (no params but harmless)
    Ignores Stage 1A heads (ECLIPSE heads do not match).
    """
    ckpt = torch.load(stage1a_ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Stage 1A keys look like: net.encoder..., net.pool..., net.proj...
    wanted_prefixes = ("net.encoder.", "net.pool.", "net.proj.", "net.act.")
    filtered = {k: v for k, v in sd.items() if k.startswith(wanted_prefixes)}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return missing, unexpected


class IELTSLightning(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        stage1a_ckpt: str,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        loss_name: str = "mse",
        warmup_ratio: float = 0.06,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = IELTSScorer(model_name=model_name)
        self.crit = build_loss(loss_name)

        # load Stage1A backbone weights
        missing, unexpected = load_stage1a_weights_into_ielts(self.net, stage1a_ckpt)
        # You can log these once if desired:
        # self.print("Loaded Stage1A. Missing:", missing[:5], "Unexpected:", unexpected[:5])

        self._train_losses = []
        self._val_losses = []

    def _masked_loss(self, preds: Dict[str, torch.Tensor], labels_list: List[Dict[str, torch.Tensor]]):
        losses = []
        for i, lab in enumerate(labels_list):
            for k, y in lab.items():
                if k not in IELTS_KEYS:
                    continue
                losses.append(self.crit(preds[k][i], y.to(preds[k].device)))
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(losses).mean()

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        preds = self.net(batch["input_ids"], batch["attention_mask"])
        loss = self._masked_loss(preds, batch["labels_list"])

        self._train_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        preds = self.net(batch["input_ids"], batch["attention_mask"])
        loss = self._masked_loss(preds, batch["labels_list"])

        self._val_losses.append(loss.detach())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero and self._train_losses:
            avg = torch.stack(self._train_losses).mean().item()
            self.print(f"\nEpoch {self.current_epoch} | Train loss: {avg:.4f}")
        self._train_losses.clear()

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero and self._val_losses:
            avg = torch.stack(self._val_losses).mean().item()
            self.print(f"Epoch {self.current_epoch} | Val   loss: {avg:.4f}")
        self._val_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
