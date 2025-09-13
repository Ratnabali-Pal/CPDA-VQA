# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from utils import dataloader_json
from dataset import VizWizDataset
from model import VQAModel

def run_epoch(model, dataloader, optimizer, criterion, is_training=True):
    """Runs a single epoch of training or validation."""
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct_ans = 0
    total_correct_type = 0
    total_samples = 0

    for (data, ans, ans_type) in tqdm(dataloader):
        data = data.to(config.DEVICE)
        ans = ans.to(config.DEVICE)
        ans_type = ans_type.to(config.DEVICE)

        with torch.set_grad_enabled(is_training):
            output, aux = model(data)
            
            loss_ans = criterion(output, ans)
            loss_type = criterion(aux, ans_type)
            loss_combined = loss_ans + loss_type
            
            if is_training:
                optimizer.zero_grad()
                loss_combined.backward()
                optimizer.step()

        total_loss += loss_combined.item()
        
        # Answer accuracy
        _, predicted_ans = torch.max(output, dim=1)
        total_correct_ans += (predicted_ans == ans).sum().item()
        
        # Type accuracy
        _, predicted_type = torch.max(aux, dim=1)
        total_correct_type += (predicted_type == ans_type).sum().item()
        
        total_samples += ans.size(0)

    avg_loss = total_loss / len(dataloader)
    acc_ans = total_correct_ans / total_samples
    acc_type = total_correct_type / total_samples
    avg_acc = (acc_ans + acc_type) / 2
    
    return avg_loss, acc_ans, acc_type, avg_acc


def main():
    # --- 1. Load and Prepare Data ---
    train_df = dataloader_json(config.ANNOTATIONS_TRAIN_PATH)
    val_df = dataloader_json(config.ANNOTATIONS_VAL_PATH)
    data_df = pd.concat((train_df, val_df), axis=0, ignore_index=True)

    # Encode labels
    ans_lb = LabelEncoder()
    data_df['answer'] = ans_lb.fit_transform(data_df['answer'])
    ans_type_lb = LabelEncoder()
    data_df['answer_type'] = ans_type_lb.fit_transform(data_df['answer_type'])

    num_classes = len(np.unique(ans_lb.classes_))
    num_aux_classes = len(np.unique(ans_type_lb.classes_))

    # Split data
    train_part_df = data_df.iloc[:len(train_df)]
    val_part_df = data_df.iloc[len(train_df):]
    
    indices = np.arange(len(train_part_df))
    train_indices, test_indices = train_test_split(indices, test_size=0.05, random_state=42, stratify=train_part_df['answer_type'])
    val_indices = np.arange(len(train_part_df), len(data_df))

    train_data = data_df.iloc[train_indices].reset_index(drop=True)
    val_data = data_df.iloc[val_indices].reset_index(drop=True)
    
    # --- 2. Load Embeddings and Create DataLoaders ---
    print("Loading pre-computed embeddings...")
    encodings = torch.load(config.ENCODINGS_PATH)

    train_dataset = VizWizDataset(encodings, train_indices, train_data['answer'].values, train_data['answer_type'].values)
    val_dataset = VizWizDataset(encodings, val_indices, val_data['answer'].values, val_data['answer_type'].values)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- 3. Initialize Model, Optimizer, and Loss ---
    model = VQAModel(
        embedding_dim=config.EMBEDDING_SIZE,
        num_classes=num_classes,
        num_aux_classes=num_aux_classes
    ).to(config.DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # --- 4. Training Loop ---
    best_val_acc = 0.0
    for epoch in range(config.EPOCHS):
        print(f"--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        train_loss, train_acc_ans, train_acc_type, avg_train_acc = run_epoch(
            model, train_dataloader, optimizer, criterion, is_training=True
        )
        print(f"Train -> Loss: {train_loss:.4f} | Ans ACC: {train_acc_ans*100:.2f}% | Type ACC: {train_acc_type*100:.2f}% | AVG ACC: {avg_train_acc*100:.2f}%")

        val_loss, val_acc_ans, val_acc_type, avg_val_acc = run_epoch(
            model, val_dataloader, None, criterion, is_training=False
        )
        print(f"Val   -> Loss: {val_loss:.4f} | Ans ACC: {val_acc_ans*100:.2f}% | Type ACC: {val_acc_type*100:.2f}% | AVG ACC: {avg_val_acc*100:.2f}%")

        # Save the best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()