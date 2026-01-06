import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from model import IELTSScorerModel
from train import IELTSLitModule
from datamodule import IELTSDataModule
from config import Config

def run_evaluation(checkpoint_path):
    # 1. Setup Device and Model
    # Detect the device (e.g., cuda:0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lit_model = IELTSLitModule.load_from_checkpoint(checkpoint_path)
    lit_model.to(device)  # Move the model to the GPU
    lit_model.eval()
    lit_model.freeze()
    
    dm = IELTSDataModule(Config.TRAIN_JSON, Config.MODEL_NAME)
    dm.setup()
    val_loader = dm.val_dataloader()
    
    results = {"TA": [], "CC": [], "LR": [], "GA": []}
    targets = {"TA": [], "CC": [], "LR": [], "GA": []}

    print(f"Evaluating on {device}...")
    
    # 2. Collect Predictions
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # MOVE DATA TO DEVICE HERE
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            extra = batch['extra_features'].to(device)
            
            # Forward pass
            logits_dict = lit_model.model(input_ids, mask, extra)
            
            for k in Config.CRITERIA:
                probas = torch.sigmoid(logits_dict[k])
                preds = (probas > 0.5).sum(dim=1) + 1
                
                # Move back to CPU for numpy/metrics processing
                results[k].extend(preds.cpu().numpy())
                targets[k].extend(batch['labels'][k].numpy()) # Labels are already on CPU

    # 3. Calculate and Print Metrics
    summary = []
    for k in Config.CRITERIA:
        qwk = cohen_kappa_score(targets[k], results[k], weights='quadratic', labels=np.arange(1, 10))
        mae = mean_absolute_error(targets[k], results[k])
        summary.append({
            "Criterion": k,
            "QWK (Alignment)": round(qwk, 4),
            "MAE (Band Error)": round(mae, 4)
        })

    # 4. Display as Table
    df = pd.DataFrame(summary)
    print("\n" + "="*40)
    print("FINAL VALIDATION RESULTS")
    print("="*40)
    print(df.to_string(index=False))
    print("="*40)
    
    # Calculate Overall average QWK
    avg_qwk = df["QWK (Alignment)"].mean()
    print(f"Mean Quadratic Weighted Kappa: {avg_qwk:.4f}")

if __name__ == "__main__":
    # Replace with your actual best checkpoint path from the logs
    BEST_CHECKPOINT = "/mount/arbeitsdaten/studenten4/rasoulta/ARES/lightning_logs/version_8/checkpoints/epoch=8-step=873.ckpt" 
    run_evaluation(BEST_CHECKPOINT)
