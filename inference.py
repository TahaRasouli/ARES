import torch
from transformers import AutoTokenizer
from model import IELTSScorerModel
from losses import logits_to_score
from utils import extract_linguistic_features
from config import Config

def score_essay(prompt, essay, checkpoint_path):
    # 1. Load Model
    model = IELTSScorerModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # 2. Preprocess
    full_text = f"PROMPT: {prompt}\n\nESSAY: {essay}"
    encoding = tokenizer(
        full_text, 
        max_length=Config.MAX_LENGTH, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    
    extra = extract_linguistic_features(essay).unsqueeze(0)
    
    # 3. Predict
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'], extra)
        
    scores = {k: logits_to_score(v).item() for k, v in logits.items()}
    
    # 4. Calculate Overall (IELTS rounds to nearest .5, but we'll return average)
    overall = sum(scores.values()) / 4
    scores["Overall"] = overall
    
    return scores

# Example Usage:
# result = score_essay("Should schools teach cooking?", "I believe cooking is vital...", "best_model.ckpt")
# print(result)
