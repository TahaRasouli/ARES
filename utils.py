import torch
import re
from textstat import flesch_reading_ease

def extract_linguistic_features(text):
    """
    Extracts 4 key features for GA and LR heads.
    """
    words = text.split()
    word_count = len(words)
    sentences = re.split(r'[.!?]+', text)
    sent_count = max(1, len([s for s in sentences if len(s.strip()) > 0]))
    
    # 1. Length feature (normalized)
    len_feat = word_count / 500.0
    
    # 2. Sentence Complexity (normalized)
    complexity = (word_count / sent_count) / 40.0
    
    # 3. Readability (Flesch Ease)
    readability = flesch_reading_ease(text) / 100.0
    
    # 4. Lexical Variety (Type-Token Ratio)
    ttr = len(set(words)) / (word_count + 1)
    
    return torch.tensor([len_feat, complexity, readability, ttr], dtype=torch.float)

def get_ordinal_labels(target, num_classes=9):
    """
    Score 7 -> [1, 1, 1, 1, 1, 1, 0, 0] (8 thresholds)
    """
    levels = torch.arange(2, num_classes + 1)
    return (target.unsqueeze(1) >= levels).float()
