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
        scores = self.attn(hidden_states).squeeze(-1)  # [B, T]

        # Use dtype-safe minimum (works in fp16/bf16/fp32)
        min_val = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(attention_mask == 0, min_val)

        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return pooled
