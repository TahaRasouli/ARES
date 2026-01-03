import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, attn_hidden: int = 256):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # token_embeddings: [B, T, H]
        # attention_mask: [B, T] (1 = keep, 0 = pad)
        scores = self.scorer(token_embeddings).squeeze(-1)  # [B, T]

        # Mask in fp32 to avoid fp16 overflow issues
        scores = scores.float().masked_fill(attention_mask == 0, torch.finfo(torch.float32).min)
        weights = torch.softmax(scores, dim=-1).to(token_embeddings.dtype)  # [B, T]

        pooled = torch.sum(token_embeddings * weights.unsqueeze(-1), dim=1)  # [B, H]
        return pooled
