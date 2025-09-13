# evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import dataloader_json
from model import Model
from train import VQADataset

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
    
    # Split data to get test indices
    indices = np.arange(len(train_df))
    _, test_indices = train_test_split(indices, test_size=0.05, random_state=42, stratify=data_df.iloc[:len(train_df)]['answer_type'])

    # Create test dataset and dataloader
    BATCH_SIZE = 64
    
    testDataset = VQADataset(encodings, test_indices, data_df.iloc[test_indices]['answer'].values, data_df.iloc[test_indices]['answer_type'].values, len(test_indices))
    test_dataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_size = encodings.shape[1]
    classes = len(np.unique(ans_lb.classes_))
    aux_classes = len(np.unique(ans_type_lb.classes_))
    
    model = Model(embedding_size, classes, aux_classes).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    # --- Evaluation ---
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0.0
    total_correct_test = 0
    total_samples_test = 0
    total_correct_test_ty = 0
    total_samples_test_ty = 0

    with torch.no_grad():
        for (data, ans, ans_type) in tqdm(test_dataloader, desc="Testing"):
            data = data.to(device)
            ans = ans.to(device)
            ans_type = ans_type.to(device)
            
            output, aux = model(data)
            
            loss_ans = loss_fn(output, ans)
            loss_type = loss_fn(aux, ans_type)
            loss_combined = loss_ans + loss_type
            test_loss += loss_combined.item()
            
            _, test_predicted = torch.max(output, dim=1)
            total_correct_test += (test_predicted == ans).sum().item()
            total_samples_test += ans.size(0)
            
            _, test_predicted_ty = torch.max(aux, dim=1)
            total_correct_test_ty += (test_predicted_ty == ans_type).sum().item()
            total_samples_test_ty += ans_type.size(0)

    test_accuracy = total_correct_test / total_samples_test
    test_accuracy_ty = total_correct_test_ty / total_samples_test_ty
    total_test_accuracy = (test_accuracy + test_accuracy_ty) / 2
    
    print(f"\nTest TYPE ACC: {test_accuracy_ty * 100:.4f}% | Test ANS ACC: {test_accuracy * 100:.4f}% | AVG Test ACC: {total_test_accuracy * 100:.4f}% | Test Loss: {test_loss / len(test_dataloader):.4f}")