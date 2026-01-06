import torch

class Config:
    # Model & Tokenizer
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 768
    
    # Data Paths
    TRAIN_JSON = "Dataset/preference_data_clean.json"
    OUTPUT_DIR = "checkpoints/"
    
    # Training Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    ACCELERATOR = "gpu"
    DEVICES = 2 # Multi-GPU setup
    
    # IELTS Specifics
    CRITERIA = ["TA", "CC", "LR", "GA"]
    NUM_CLASSES = 9  # Bands 1-9
    NUM_EXTRA_FEATURES = 4 # From utils.py
