import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [B, T, H]
        attention_mask: [B, T]
        """
        scores = self.attn(hidden_states).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return pooled  # [B, H]

