import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from ielts_datamodule import IELTSDataModule
from ielts_lit_module import IELTSLightning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task1_jsonl", type=str, required=True)
    ap.add_argument("--task2_json", type=str, required=True)
    ap.add_argument("--stage1a_ckpt", type=str, required=True)

    ap.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "huber"])
    ap.add_argument("--warmup_ratio", type=float, default=0.06)

    ap.add_argument("--precision", type=str, default="16-mixed")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    dm = IELTSDataModule(
        task1_jsonl_path=args.task1_jsonl,
        task2_json_path=args.task2_json,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = IELTSLightning(
        model_name=args.model_name,
        stage1a_ckpt=args.stage1a_ckpt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        warmup_ratio=args.warmup_ratio,
    )

    ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="stage1b-ielts-{epoch:02d}-{val_loss:.4f}",
    )
    lrmon = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp_find_unused_parameters_true",  # TA vs TR heads not always used
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=[ckpt, lrmon],
        log_every_n_steps=20,
        enable_progress_bar=False,   # keeps prints from disappearing
        num_sanity_val_steps=0,      # avoids the double val print at epoch 0
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
