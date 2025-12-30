import pytorch_lightning as pl
from transformers import AutoTokenizer
from datamodule import EclipseDataModule
from lightning_module import EclipseLightningModule

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-base"
    )

    datamodule = EclipseDataModule(
        csv_path="ECLIPSE_train.csv",
        tokenizer=tokenizer,
        batch_size=4,   # per-GPU batch size
        max_len=512
    )

    model = EclipseLightningModule(lr=2e-5)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=2,                  # 👈 TWO GPUs
        strategy="ddp",              # 👈 Distributed Data Parallel
        precision="16-mixed",
        log_every_n_steps=20,
        enable_checkpointing=True
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
