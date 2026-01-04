import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging

from task2_datamodule import Task2DataModule
from task2_lit_module import Task2Lightning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task2_path", type=str, required=True)
    ap.add_argument("--init_ckpt", type=str, required=True)

    ap.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=768)  # bump default
    ap.add_argument("--epochs", type=int, default=15)

    ap.add_argument("--lr_encoder", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--loss", type=str, default="huber", choices=["mse", "mae", "huber"])
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--layerwise_decay", type=float, default=0.95)

    ap.add_argument("--precision", type=str, default="16-mixed")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--enforce_half_bands", action="store_true")
    ap.add_argument("--max_gap", type=float, default=4.0)

    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)

    dm = Task2DataModule(
        task2_path=args.task2_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        enforce_half_bands=args.enforce_half_bands,
        max_gap=args.max_gap,
    )

    model = Task2Lightning(
        model_name=args.model_name,
        init_ckpt=args.init_ckpt,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        warmup_ratio=args.warmup_ratio,
        layerwise_decay=args.layerwise_decay,
    )

    ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="task2-stage2-best-{epoch:02d}-{val_loss:.4f}",
    )
    lrmon = LearningRateMonitor(logging_interval="step")

    # SWA over last ~20% of epochs
    swa = StochasticWeightAveraging(swa_lrs=args.lr_encoder, swa_epoch_start=0.8)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=[ckpt, lrmon, swa],
        enable_progress_bar=False,       # epoch prints persist
        enable_model_summary=False,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,           # stabilizes finetuning
        deterministic=False,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
