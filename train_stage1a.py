import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from eclipse_datamodule import EclipseDataModule
from eclipse_lit_module import EclipseLightning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--precision", default="16-mixed")

    args = ap.parse_args()

    dm = EclipseDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    model = EclipseLightning(
        model_name=args.model_name,
        lr=args.lr,
    )

    ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="stage1a-eclipse-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=[ckpt],
        enable_progress_bar=False,  # ensures prints persist
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
