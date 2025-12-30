import torch.nn as nn
from transformers import AutoModel
from pooling import AttentionPooling

class EssayEncoder(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768

        self.pooling = AttentionPooling(hidden)

        self.representation_head = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = self.pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        return self.representation_head(pooled)  # [B, 512]
