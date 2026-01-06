import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from model import IELTSScorerModel
from train import IELTSLitModule
from datamodule import IELTSDataModule
from config import Config
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def run_evaluation(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = IELTSLitModule.load_from_checkpoint(checkpoint_path)
    lit_model.to(device)
    lit_model.eval()
    
    dm = IELTSDataModule(Config.TRAIN_JSON, Config.MODEL_NAME)
    dm.setup()
    val_loader = dm.val_dataloader()
    
    # Initialize dictionaries with empty lists
    results = {k: [] for k in Config.CRITERIA}
    targets = {k: [] for k in Config.CRITERIA}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            extra = batch['extra_features'].to(device)
            
            logits_dict = lit_model.model(input_ids, mask, extra)
            
            for k in Config.CRITERIA:
                # 1. Convert logits to probabilities
                probas = torch.sigmoid(logits_dict[k])
                # 2. Ordinal logic: Sum threshold crossings + 1
                preds = (probas > 0.5).sum(dim=1) + 1
                
                # 3. Explicitly move to CPU and convert to list
                results[k].extend(preds.cpu().tolist())
                targets[k].extend(batch['labels'][k].tolist())

    summary = []
    for k in Config.CRITERIA:
        y_true = np.array(targets[k])
        y_pred = np.array(results[k])
        
        # FIX: Check if we actually have data
        if len(y_true) == 0:
            print(f"Warning: No data collected for criterion {k}!")
            continue

        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=np.arange(1, 10))
        mae = mean_absolute_error(y_true, y_pred)
        
        summary.append({
            "Criterion": k,
            "QWK": round(qwk, 4),
            "MAE": round(mae, 4)
        })

    df = pd.DataFrame(summary)
    print("\n", df.to_string(index=False))

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

def plot_confusion_matrices(targets, results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, k in enumerate(Config.CRITERIA):
        cm = confusion_matrix(targets[k], results[k], labels=np.arange(1, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=np.arange(1, 10), yticklabels=np.arange(1, 10))
        axes[i].set_title(f'Confusion Matrix: {k}')
        axes[i].set_xlabel('Predicted Band')
        axes[i].set_ylabel('True Band')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("Confusion matrices saved to confusion_matrices.png")

if __name__ == "__main__":
    # Replace with your actual best checkpoint path from the logs
    BEST_CHECKPOINT = "/mount/arbeitsdaten/studenten4/rasoulta/ARES/lightning_logs/version_8/checkpoints/epoch=8-step=873.ckpt" 
    run_evaluation(BEST_CHECKPOINT)
    plot_confusion_matrices(targets, results)
