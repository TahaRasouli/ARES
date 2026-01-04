import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from task2_datamodule import Task2DataModule
from task2_lit_module import Task2Lightning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task2_path", type=str, required=True)
    ap.add_argument("--init_ckpt", type=str, required=True)

    ap.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "huber"])
    ap.add_argument("--warmup_ratio", type=float, default=0.06)

    ap.add_argument("--precision", type=str, default="16-mixed")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require_all_four", action="store_true")

    args = ap.parse_args()

    dm = Task2DataModule(
        task2_path=args.task2_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        require_all_four=True if args.require_all_four else True,  # default True
    )

    model = Task2Lightning(
        model_name=args.model_name,
        init_ckpt=args.init_ckpt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        warmup_ratio=args.warmup_ratio,
    )

    ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="task2-stage2-{epoch:02d}-{val_loss:.4f}",
    )
    lrmon = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=[ckpt, lrmon],
        log_every_n_steps=50,
        enable_progress_bar=False,     # IMPORTANT: keep epoch prints from disappearing
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
