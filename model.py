import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class IELTSScorerModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_extra_features=4):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        
        hidden_size = self.config.hidden_size
        self.num_extra = num_extra_features
        
        # Project 4 concatenated layers (768*4) back to 512
        self.pooler = nn.Linear(hidden_size * 4, 512)
        self.dropout = nn.Dropout(0.1)

        # Ordinal Heads: 8 nodes for bands 2, 3, 4, 5, 6, 7, 8, 9
        self.head_ta = nn.Linear(512, 8)
        self.head_cc = nn.Linear(512, 8)
        self.head_lr = nn.Linear(512 + self.num_extra, 8) # Extra features for Lexical
        self.head_ga = nn.Linear(512 + self.num_extra, 8) # Extra features for Grammar

    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Suggestion 3: Multi-Layer Extraction (Last 4 layers)
        # hidden_states is a tuple of 13 tensors (embedding + 12 layers)
        all_layers = outputs.hidden_states
        cls_4 = torch.cat([all_layers[-i][:, 0, :] for i in range(1, 5)], dim=-1)
        
        pooled = torch.tanh(self.pooler(cls_4))
        pooled = self.dropout(pooled)
        
        # Suggestion 2: Injecting Handcrafted Features
        # Combine pooled embeddings with error counts/readability for LR and GA
        combined_features = torch.cat([pooled, extra_features], dim=1)
        
        return {
            "TA": self.head_ta(pooled),
            "CC": self.head_cc(pooled),
            "LR": self.head_lr(combined_features),
            "GA": self.head_ga(combined_features)
        }
