import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from model import IELTSScorerModel
from datamodule import IELTSDataModule
from losses import OrdinalCORALLoss, logits_to_score
import torch.nn as nn
import torch

class IELTSLitModule(L.LightningModule):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = IELTSScorerModel()
        self.loss_fn = OrdinalCORALLoss(num_classes=9)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch['input_ids'], batch['attention_mask'], batch['extra_features'])
        
        # Calculate loss for each criterion
        loss_ta = self.loss_fn(outputs['TA'], batch['labels']['TA'])
        loss_cc = self.loss_fn(outputs['CC'], batch['labels']['CC'])
        loss_lr = self.loss_fn(outputs['LR'], batch['labels']['LR'])
        loss_ga = self.loss_fn(outputs['GA'], batch['labels']['GA'])
        
        total_loss = (loss_ta + loss_cc + loss_lr + loss_ga) / 4
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch['input_ids'], batch['attention_mask'], batch['extra_features'])
        
        # Log integer-rounded accuracy
        pred_ga = logits_to_score(outputs['GA'])
        mae = torch.abs(pred_ga - batch['labels']['GA']).float().mean()
        self.log("val_ga_mae", mae, prog_bar=True)

    def configure_optimizers(self):
        # Suggestion: Discriminative Learning Rates
        # Encoder gets lower LR, Heads get higher LR
        optimizer = torch.optim.AdamW([
            {'params': self.model.encoder.parameters(), 'lr': self.hparams.lr},
            {'params': self.model.head_ta.parameters(), 'lr': self.hparams.lr * 5},
            {'params': self.model.head_ga.parameters(), 'lr': self.hparams.lr * 5},
        ])
        return optimizer

if __name__ == "__main__":
    dm = IELTSDataModule("Dataset/preference_data_clean.json", "microsoft/deberta-v3-base")
    module = IELTSLitModule()
    
    checkpoint_callback = ModelCheckpoint(monitor="val_ga_mae", mode="min")
    
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        precision="16-mixed",
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(module, datamodule=dm)
