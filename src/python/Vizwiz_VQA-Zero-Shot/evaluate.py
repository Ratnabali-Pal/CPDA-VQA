# evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import config
from utils import dataloader_json
from dataset import VizWizDataset
from model import VQAModel
from train import run_epoch # We can reuse the run_epoch function

def evaluate_model():
    # --- 1. Load and Prepare Data (similar to train.py) ---
    train_df = dataloader_json(config.ANNOTATIONS_TRAIN_PATH)
    data_df = pd.concat((train_df,), axis=0, ignore_index=True) # Only need train for split

    # Recreate label encoders to ensure consistency
    ans_lb = LabelEncoder()
    data_df['answer'] = ans_lb.fit_transform(data_df['answer'])
    ans_type_lb = LabelEncoder()
    data_df['answer_type'] = ans_type_lb.fit_transform(data_df['answer_type'])

    num_classes = len(np.unique(ans_lb.classes_))
    num_aux_classes = len(np.unique(ans_type_lb.classes_))
    
    # Get test indices from the original split
    indices = np.arange(len(data_df))
    _, test_indices = train_test_split(indices, test_size=0.05, random_state=42, stratify=data_df['answer_type'])
    
    test_data = data_df.iloc[test_indices].reset_index(drop=True)

    # --- 2. Load Embeddings and Create DataLoader ---
    print("Loading pre-computed embeddings...")
    encodings = torch.load(config.ENCODINGS_PATH)

    test_dataset = VizWizDataset(encodings, test_indices, test_data['answer'].values, test_data['answer_type'].values)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # --- 3. Load Model and Evaluate ---
    model = VQAModel(
        embedding_dim=config.EMBEDDING_SIZE,
        num_classes=num_classes,
        num_aux_classes=num_aux_classes
    ).to(config.DEVICE)
    
    # Handle DataParallel model loading
    # Create a new state_dict without the 'module.' prefix
    state_dict = torch.load(config.MODEL_SAVE_PATH)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
    print("Model loaded successfully.")

    criterion = nn.CrossEntropyLoss()
    
    print("Evaluating on the test set...")
    test_loss, test_acc_ans, test_acc_type, avg_test_acc = run_epoch(
        model, test_dataloader, None, criterion, is_training=False
    )
    
    print("\n--- Test Results ---")
    print(f"Loss: {test_loss:.4f}")
    print(f"Answer Accuracy: {test_acc_ans*100:.2f}%")
    print(f"Answer Type Accuracy: {test_acc_type*100:.2f}%")
    print(f"Average Accuracy: {avg_test_acc*100:.2f}%")

if __name__ == "__main__":
    evaluate_model()