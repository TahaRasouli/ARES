import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class IELTSScorerModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_extra_features=4):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True # Crucial for multi-layer access
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        
        hidden_size = self.config.hidden_size
        
        # We now use two separate projection layers
        self.semantic_pooler = nn.Linear(hidden_size * 4, 512)
        self.syntax_pooler = nn.Linear(hidden_size * 4, 512) # Must be same size!
        self.dropout = nn.Dropout(0.1)

        # Ordinal Heads
        self.head_ta = nn.Linear(512, 8)
        self.head_cc = nn.Linear(512, 8)
        self.head_lr = nn.Linear(512 + num_extra_features, 8)
        self.head_ga = nn.Linear(512 + num_extra_features, 8)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_layers = outputs.hidden_states # List of all transformer layers
        
        # 1. Semantic Features (Last 4 layers: 9, 10, 11, 12)
        cls_semantic = torch.cat([all_layers[-i][:, 0, :] for i in range(1, 5)], dim=-1)
        pooled_sem = torch.tanh(self.semantic_pooler(cls_semantic))
        
        # 2. Syntax Features (Middle layers: 5, 6, 7, 8) - Better for GA
        cls_syntax = torch.cat([all_layers[i][:, 0, :] for i in range(5, 9)], dim=-1)
        pooled_syn = torch.tanh(self.syntax_pooler(cls_syntax))
        
        pooled_sem = self.dropout(pooled_sem)
        pooled_syn = self.dropout(pooled_syn)
        
        # Combine with handcrafted features for specialized heads
        lr_input = torch.cat([pooled_sem, extra_features], dim=1)
        ga_input = torch.cat([pooled_syn, extra_features], dim=1)
        
        return {
            "TA": self.head_ta(pooled_sem),
            "CC": self.head_cc(pooled_sem),
            "LR": self.head_lr(lr_input),
            "GA": self.head_ga(ga_input)
        }