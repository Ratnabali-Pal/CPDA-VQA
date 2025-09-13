# train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import dataloader_json, plot_loss
from model import Model

# --- Dataset Class ---
class VQADataset(Dataset):
    def __init__(self, encodings, indices, answers, types, length):
        self.encodings = encodings
        self.indices = indices
        self.answers = answers
        self.types = types
        self.length = length

    def __getitem__(self, index):
        if self.length <= 20523:
            return self.encodings[self.indices[index]].float(), torch.tensor(int(self.answers[index])), torch.tensor(int(self.types[index]))
        
        return self.encodings[self.indices[index]].float(), torch.tensor(int(self.answers[index % (self.length // 2)])), torch.tensor(int(self.types[index % (self.length // 2)]))
            
    def __len__(self):
        return self.length

# --- Training and Validation Function ---
def run_model(model, dataloader, val_loader, optimizer, device):
    model.train()
    
    loss_fn = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_correct_ty = 0
    total_samples_ty = 0
    
    for (data, ans, ans_type) in tqdm(dataloader, desc="Training"):
        data = data.to(device)
        ans = ans.to(device)
        ans_type = ans_type.to(device)
        
        optimizer.zero_grad()
        output, aux = model(data)
        
        loss_ans = loss_fn(output, ans)
        loss_type = loss_fn(aux, ans_type)
        loss_combined = loss_ans + loss_type
        
        loss_combined.backward()
        optimizer.step()
        
        total_loss += loss_combined.item()
        
        _, predicted_labels = torch.max(output, dim=1)
        total_correct += (predicted_labels == ans).sum().item()
        total_samples += ans.size(0)
        
        _, predicted_labels_ty = torch.max(aux, dim=1)
        total_correct_ty += (predicted_labels_ty == ans_type).sum().item()
        total_samples_ty += ans_type.size(0)

    train_accuracy = total_correct / total_samples
    train_accuracy_ty = total_correct_ty / total_samples_ty
    total_train_accuracy = (train_accuracy + train_accuracy_ty) / 2
    train_loss = total_loss / len(dataloader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    total_correct_val = 0
    total_samples_val = 0
    total_correct_val_ty = 0
    total_samples_val_ty = 0
    
    with torch.no_grad():
        for (data, ans, ans_type) in tqdm(val_loader, desc="Validating"):
            data = data.to(device)
            ans = ans.to(device)
            ans_type = ans_type.to(device)
            
            output, aux = model(data)
            
            loss_ans = loss_fn(output, ans)
            loss_type = loss_fn(aux, ans_type)
            loss_combined = loss_ans + loss_type
            val_loss += loss_combined.item()
            
            _, val_predicted = torch.max(output, dim=1)
            total_correct_val += (val_predicted == ans).sum().item()
            total_samples_val += ans.size(0)
            
            _, val_predicted_ty = torch.max(aux, dim=1)
            total_correct_val_ty += (val_predicted_ty == ans_type).sum().item()
            total_samples_val_ty += ans_type.size(0)

    val_accuracy = total_correct_val / total_samples_val
    val_accuracy_ty = total_correct_val_ty / total_samples_val_ty
    total_val_accuracy = (val_accuracy + val_accuracy_ty) / 2
    val_loss /= len(val_loader)
    
    print(f"\nTrain Loss: {train_loss:.4f} | AVG Train ACC: {total_train_accuracy * 100:.4f}% | Val Loss: {val_loss:.4f} | AVG Val ACC: {total_val_accuracy * 100:.2f}%")
    print(f"Train ANS ACC: {train_accuracy * 100:.4f}% | VAL ANS ACC: {val_accuracy * 100:.4f}% | Train TYPE ACC: {train_accuracy_ty * 100:.4f}% | VAL TYPE ACC: {val_accuracy_ty * 100:.2f}%\n")
    
    return train_loss, val_loss

# --- Main Execution ---
if __name__ == '__main__':
    # Load and preprocess data
    train_df = dataloader_json("dataset/Annotations/Annotations/train.json")
    val_df = dataloader_json("dataset/Annotations/Annotations/val.json")
    data_df = pd.concat((train_df, val_df), axis=0, ignore_index=True)

    ans_lb = LabelEncoder()
    data_df['answer'] = ans_lb.fit_transform(data_df['answer'])
    ans_type_lb = LabelEncoder()
    data_df['answer_type'] = ans_type_lb.fit_transform(data_df['answer_type'])

    # Load encodings
    encodings = torch.cat(torch.load("withgpt.pt"))
    
    # Split data
    indices = np.arange(len(train_df))
    train_indices, _ = train_test_split(indices, test_size=0.05, random_state=42, stratify=data_df.iloc[:len(train_df)]['answer_type'])
    train_indices_aug = train_indices + len(train_df)
    
    val_indices = np.arange(len(train_df), len(data_df))

    # Create datasets and dataloaders
    BATCH_SIZE = 64
    
    trainDataset = VQADataset(encodings, np.concatenate((train_indices, train_indices_aug)), data_df.iloc[train_indices]['answer'].values, data_df.iloc[train_indices]['answer_type'].values, len(train_indices) * 2)
    valDataset = VQADataset(encodings, val_indices, data_df.iloc[val_indices]['answer'].values, data_df.iloc[val_indices]['answer_type'].values, len(val_df))
    
    train_dataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model, optimizer, scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_size = encodings.shape[1]
    classes = len(np.unique(ans_lb.classes_))
    aux_classes = len(np.unique(ans_type_lb.classes_))
    
    model = Model(embedding_size, classes, aux_classes).to(device)
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, threshold=1e-6)

    # Training loop
    epochs = 125
    training_loss = []
    val_loss_list = []
    
    early_stopping_patience = 15
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for e in range(epochs):
        print(f'Epoch: {e+1} | LR: {optimizer.param_groups[0]["lr"]}')
        trLoss, vlLoss = run_model(model, train_dataloader, val_dataloader, optimizer, device)    
        training_loss.append(trLoss)
        val_loss_list.append(vlLoss)
        
        scheduler.step(vlLoss)
        
        if vlLoss < best_val_loss:
            best_val_loss = vlLoss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved new best model.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nValidation loss hasn't improved for {early_stopping_patience} epochs. Early stopping.")
            break
            
    plot_loss(training_loss, val_loss_list)