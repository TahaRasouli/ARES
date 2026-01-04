from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers.optimization import get_linear_schedule_with_warmup

from task2_model import Task2Scorer
from task2_data import TASK2_KEYS


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError("loss must be one of: mse, mae, huber")


def load_init_from_ckpt(model: nn.Module, ckpt_path: str) -> Tuple[int, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    allowed_prefixes = ("net.encoder.", "net.pool.", "net.proj.")
    allowed_heads = tuple(f"net.heads.{k}." for k in TASK2_KEYS)

    filtered = {}
    for k, v in sd.items():
        if k.startswith(allowed_prefixes) or k.startswith(allowed_heads):
            filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return len(missing), len(unexpected)


def iter_encoder_layers(hf_model):
    """
    DeBERTa-v3 (DebertaV2Model) exposes encoder layers at:
      model.encoder.layer
    """
    enc = getattr(hf_model, "encoder", None)
    if enc is None:
        return []
    layers = getattr(enc, "layer", None)
    if layers is None:
        return []
    return list(layers)


class Task2Lightning(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        init_ckpt: str,
        lr_encoder: float = 1e-5,
        lr_head: float = 5e-5,
        weight_decay: float = 0.01,
        loss_name: str = "huber",
        warmup_ratio: float = 0.06,
        layerwise_decay: float = 0.95,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = Task2Scorer(model_name=model_name)
        self.crit = build_loss(loss_name)

        miss, unexp = load_init_from_ckpt(self.net, init_ckpt)
        self._init_missing = miss
        self._init_unexpected = unexp

        self._train_losses = []
        self._val_losses = []

        # accumulate per-criterion sums for RMSE
        self._val_se = {k: [] for k in TASK2_KEYS}  # squared errors

    def on_fit_start(self):
        if self.trainer.is_global_zero:
            self.print(f"Init checkpoint loaded. missing_keys={self._init_missing} unexpected_keys={self._init_unexpected}")

    def _masked_loss(self, preds: Dict[str, torch.Tensor], labels_list: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        losses = []
        for i, lab in enumerate(labels_list):
            for k, y in lab.items():
                if k in TASK2_KEYS:
                    losses.append(self.crit(preds[k][i], y.to(preds[k].device)))
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(losses).mean()

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        preds = self.net(batch["input_ids"], batch["attention_mask"])
        loss = self._masked_loss(preds, batch["labels_list"])
        self._train_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        preds = self.net(batch["input_ids"], batch["attention_mask"])
        loss = self._masked_loss(preds, batch["labels_list"])
        self._val_losses.append(loss.detach())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # compute per-criterion squared error on ORIGINAL band scale:
        # labels and preds are in [0,1] => convert to [0,9] by *9
        for i, lab in enumerate(batch["labels_list"]):
            for k, y in lab.items():
                if k not in TASK2_KEYS:
                    continue
                yb = y.to(preds[k].device) * 9.0
                pb = preds[k][i] * 9.0
                se = (pb - yb) ** 2
                self._val_se[k].append(se.detach())

        return loss

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero and self._train_losses:
            avg = torch.stack(self._train_losses).mean().item()
            # convert train loss back to band-scale RMSE approx for intuition:
            # if loss is MSE on normalized scale, RMSE_band ~ sqrt(loss)*9
            self.print(f"\nEpoch {self.current_epoch} | Train loss: {avg:.4f}")
        self._train_losses.clear()

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero and self._val_losses:
            avg = torch.stack(self._val_losses).mean().item()
            self.print(f"Epoch {self.current_epoch} | Val   loss: {avg:.4f}")

            # per-criterion RMSE (band scale)
            rmse_parts = []
            for k in TASK2_KEYS:
                if self._val_se[k]:
                    mse_k = torch.stack(self._val_se[k]).mean().item()
                    rmse_k = mse_k ** 0.5
                    rmse_parts.append(f"{k}={rmse_k:.3f}")
                self._val_se[k].clear()

            if rmse_parts:
                self.print(f"Epoch {self.current_epoch} | Val RMSE (bands): " + " | ".join(rmse_parts))

        self._val_losses.clear()

    def configure_optimizers(self):
        # ---- Discriminative LR + layer-wise decay for encoder ----
        encoder = self.net.encoder

        # Identify encoder layers (top to bottom)
        layers = iter_encoder_layers(encoder)
        n_layers = len(layers)

        param_groups = []

        # Head/pool/proj get higher LR
        head_params = list(self.net.pool.parameters()) + list(self.net.proj.parameters()) + list(self.net.heads.parameters())
        param_groups.append({"params": head_params, "lr": self.hparams.lr_head, "weight_decay": self.hparams.weight_decay})

        # Encoder embedding + lower components
        # Start with base lr for bottom, increase slightly toward top
        base_lr = self.hparams.lr_encoder
        decay = self.hparams.layerwise_decay

        # Add embeddings + everything not in layer blocks separately
        # (This is a simple catch-all: parameters not included later will be added here)
        handled = set()
        for p in head_params:
            handled.add(id(p))

        # Layer blocks
        for idx, layer in enumerate(layers):
            # idx=0 is bottom, idx=n_layers-1 is top
            layer_lr = base_lr * (decay ** (n_layers - 1 - idx))
            params = [p for p in layer.parameters() if p.requires_grad]
            for p in params:
                handled.add(id(p))
            param_groups.append({"params": params, "lr": layer_lr, "weight_decay": self.hparams.weight_decay})

        # Remaining encoder params (embeddings, norms, etc.)
        rest = []
        for p in encoder.parameters():
            if p.requires_grad and id(p) not in handled:
                rest.append(p)
        if rest:
            param_groups.append({"params": rest, "lr": base_lr, "weight_decay": self.hparams.weight_decay})

        optimizer = torch.optim.AdamW(param_groups)

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
