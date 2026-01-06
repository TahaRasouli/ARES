import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalCORALLoss(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        """
        logits: [batch, num_classes - 1]
        targets: [batch] (integers from 1 to 9)
        """
        # Convert targets to cumulative binary vectors
        # e.g., Score 7 -> [1, 1, 1, 1, 1, 1, 0, 0]
        num_thresholds = self.num_classes - 1
        device = logits.device
        
        # levels: [1, num_thresholds] -> [2, 3, 4, 5, 6, 7, 8, 9]
        levels = torch.arange(2, self.num_classes + 1, device=device).view(1, -1)
        # targets: [batch, 1]
        binary_labels = (targets.view(-1, 1) >= levels).float()
        
        # Compute BCE loss across all threshold nodes
        loss = F.binary_cross_entropy_with_logits(logits, binary_labels)
        return loss

def logits_to_score(logits):
    """Converts threshold probabilities to an integer band 1-9."""
    probas = torch.sigmoid(logits)
    # Sum of probabilities + 1 (base band)
    return (probas > 0.5).sum(dim=1) + 1
