import pytorch_lightning as L

class IELTSLoggingCallback(L.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Access logged metrics
        metrics = trainer.callback_metrics
        if "val_ga_mae" in metrics:
            print(f"\n--- Epoch {trainer.current_epoch} Summary ---")
            print(f"Validation Loss: {metrics['val_loss']:.4f}")
            print(f"GA Integer Error (MAE): {metrics['val_ga_mae']:.2f} bands")
