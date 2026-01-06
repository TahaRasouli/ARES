import torch
from torch.utils.data import Dataset
from utils import extract_ga_refined_features # Import your new function
import json
import re

class IELTSDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=768):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _extract_features(self, text):
        # Placeholder for handcrafted features
        # In a real setup, use libraries like 'textstat' or 'language-tool-python'
        words = text.split()
        word_count = len(words)
        sent_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        unique_words = len(set(words))
        
        # Normalized features
        avg_sent_len = word_count / sent_count
        lex_diversity = unique_words / (word_count + 1)
        
        return torch.tensor([word_count/500, avg_sent_len/30, lex_diversity, sent_count/20], dtype=torch.float)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"PROMPT: {item['prompt']}\n\nESSAY: {item['essay']}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Use the refined GA features here
        extra = extract_ga_refined_features(item['essay'])
        
        return {
            "input_ids": encoding['input_ids'].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "extra_features": extra, # This now contains complexity/readability/sent_len
            "labels": {
                "TA": torch.tensor(item['TA'], dtype=torch.long),
                "CC": torch.tensor(item['CC'], dtype=torch.long),
                "LR": torch.tensor(item['LR'], dtype=torch.long),
                "GA": torch.tensor(item['GA'], dtype=torch.long),
            }
        }
        
        return {
            "input_ids": encoding['input_ids'].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "extra_features": extra,
            "labels": {
                "TA": torch.tensor(item['TA'], dtype=torch.long),
                "CC": torch.tensor(item['CC'], dtype=torch.long),
                "LR": torch.tensor(item['LR'], dtype=torch.long),
                "GA": torch.tensor(item['GA'], dtype=torch.long),
            }
        }
