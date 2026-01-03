import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datamodule import EssayDataModule
from lit_module import EssayLightning


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--task1_jsonl", type=str, required=True)
    ap.add_argument("--task2_json", type=str, required=True)
    ap.add_argument("--eclipse_train_csv", type=str, required=True)
    ap.add_argument("--eclipse_val_csv", type=str, required=True)

    ap.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adafactor", "lion"])
    ap.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "huber"])
    ap.add_argument("--warmup_ratio", type=float, default=0.06)

    ap.add_argument("--precision", type=str, default="16-mixed")
    ap.add_argument("--num_workers", type=int, default=4)

    args = ap.parse_args()

    dm = EssayDataModule(
        task1_jsonl_path=args.task1_jsonl,
        task2_json_path=args.task2_json,
        eclipse_train_csv=args.eclipse_train_csv,
        eclipse_val_csv=args.eclipse_val_csv,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    model = EssayLightning(
        model_name=args.model_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        loss_name=args.loss,
        warmup_ratio=args.warmup_ratio,
    )

    ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )
    lrmon = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=[ckpt, lrmon],
        log_every_n_steps=20,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
