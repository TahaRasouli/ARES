from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModel

from dataset_unified import ALL_TARGET_KEYS
from pooling import AttentionPooling


class EssayScorer(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", proj_dim: int = 512):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.pool = AttentionPooling(hidden_size=hidden, attn_hidden=256)
        self.proj = nn.Linear(hidden, proj_dim)
        self.act = nn.GELU()

        self.heads = nn.ModuleDict({k: nn.Linear(proj_dim, 1) for k in ALL_TARGET_KEYS})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state  # [B, T, H]

        pooled = self.pool(token_emb, attention_mask)  # [B, H]
        z = self.act(self.proj(pooled))  # [B, D]

        preds = {k: self.heads[k](z).squeeze(-1) for k in self.heads.keys()}  # each [B]
        return preds
