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

    # Inside your IELTSScorerModel forward pass
    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_layers = outputs.hidden_states
        
        # TA/CC/LR use the last 4 layers (Semantic layers)
        cls_semantic = torch.cat([all_layers[-i][:, 0, :] for i in range(1, 5)], dim=-1)
        pooled_semantic = torch.tanh(self.pooler(cls_semantic))
        
        # GA uses "Syntax Layers" (e.g., layers 6, 7, 8, 9)
        # This helps catch subject-verb agreement and tense issues
        cls_syntax = torch.cat([all_layers[i][:, 0, :] for i in range(6, 10)], dim=-1)
        pooled_syntax = torch.tanh(self.pooler(cls_syntax)) # Re-using pooler or defining a second one
        
        ga_input = torch.cat([pooled_syntax, extra_features], dim=1)
        
        return {
            "TA": self.head_ta(pooled_semantic),
            "CC": self.head_cc(pooled_semantic),
            "LR": self.head_lr(torch.cat([pooled_semantic, extra_features], dim=1)),
            "GA": self.head_ga(ga_input)
        }