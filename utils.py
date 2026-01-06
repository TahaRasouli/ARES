import torch
import re
from textstat import sentence_count, automated_readability_index

def extract_ga_refined_features(text):
    words = text.split()
    word_count = max(1, len(words))
    sents = sentence_count(text)
    
    # 1. Complexity Metric (Range): Ratio of long sentences
    # IELTS Band 7+ requires 'complex structures'
    sentences = re.split(r'[.!?]+', text)
    long_sentences = sum(1 for s in sentences if len(s.split()) > 20)
    complexity_ratio = long_sentences / max(1, sents)
    
    # 2. Accuracy Proxy: Readability Index
    # Lower readability often correlates with awkward grammar/syntax
    ari = automated_readability_index(text) / 20.0  # Keep this normalized!
    avg_sent_len = (word_count / max(1, sents)) / 50.0 # Standardize scale
    return torch.tensor([complexity_ratio, ari, avg_sent_len, 1.0], dtype=torch.float)
    
    return torch.tensor([complexity_ratio, ari, avg_sent_len, 1.0], dtype=torch.float)

def get_ordinal_labels(target, num_classes=9):
    """
    Score 7 -> [1, 1, 1, 1, 1, 1, 0, 0] (8 thresholds)
    """
    levels = torch.arange(2, num_classes + 1)
    return (target.unsqueeze(1) >= levels).float()
