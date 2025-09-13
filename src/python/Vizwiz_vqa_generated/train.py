import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm
import clip
from PIL import Image

from data_loader import dataloader_json, plot_loss
from dataset import VizWizDataset
from model import VQAModel

def run_model(model, dataloader, val_loader, optimizer, device):
    model.train()
    
    loss_fn = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    total_correct_ty = 0
    total_samples_ty = 0
    
    for (data, ans, ans_type) in tqdm(dataloader):
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

    # Validation
    model.eval()
    val_loss = 0.0
    total_correct_val = 0
    total_samples_val = 0
    total_correct_val_ty = 0
    total_samples_val_ty = 0

    with torch.no_grad():
        for (data, ans, ans_type) in val_loader:
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
    val_loss = val_loss / len(val_loader)

    print(f"\nTrain Loss: {train_loss:.4f} | AVG Train ACC: {total_train_accuracy * 100:.4f}% | Val Loss: {val_loss:.4f} | AVG Val ACC: {total_val_accuracy * 100:.2f}%")
    print(f"Train ANS ACC: {train_accuracy * 100:.4f}% | VAL ANS ACC: {val_accuracy * 100:.4f}% | Train TYPE ACC: {train_accuracy_ty * 100:.4f}% | VAL TYPE ACC: {val_accuracy_ty * 100:.2f}%\n")
    
    return train_loss, val_loss

def main():
    # Note: You need to download the VizWiz dataset and place it in a 'dataset' directory.
    # The expected structure is:
    # dataset/
    #   Annotations/
    #       Annotations/
    #           generated.json
    #           val.json
    #   train/
    #       train/
    #           ... (images)
    #   val/
    #       val/
    #           ... (images)
    #   test/
    #       test/
    #           ... (images)

    try:
        train_df = dataloader_json("dataset/Annotations/Annotations/generated.json")
        val_df = dataloader_json("dataset/Annotations/Annotations/val.json")
    except FileNotFoundError:
        print("Error: Annotation files not found. Please make sure the dataset is in the correct directory structure.")
        return

    data_df = pd.concat((train_df, val_df), axis=0, ignore_index=True)

    ans_lb = LabelEncoder()
    data_df['answer'] = ans_lb.fit_transform(data_df['answer'])
    ans_type_lb = LabelEncoder()
    data_df['answer_type'] = ans_type_lb.fit_transform(data_df['answer_type'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device}')
    
    try:
        model_clip, preprocess = clip.load("ViT-L/14", device=device)
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        print("Please ensure you have an internet connection and the necessary dependencies installed.")
        return

    # Generate encodings if they don't exist
    try:
        encodings_tensor = torch.load('encodings.pt')
        print("Loaded pre-computed encodings.")
    except FileNotFoundError:
        print("Pre-computed encodings not found. Generating new encodings...")
        encodings = []
        for img_path, question in tqdm(zip(data_df['image'], data_df['question']), total=len(data_df)):
            if "train" in img_path:
                image_full_path = f'dataset/train/train/{img_path}'
            elif "val" in img_path:
                image_full_path = f'dataset/val/val/{img_path}'
            else:
                image_full_path = f'dataset/test/test/{img_path}'
            
            try:
                image = preprocess(Image.open(image_full_path)).unsqueeze(0).to(device)
                text = clip.tokenize(question).to(device)
                with torch.no_grad():
                    image_encoding = model_clip.encode_image(image)
                    text_encoding = model_clip.encode_text(text)
                encodings.append(torch.cat([image_encoding, text_encoding], dim=-1))
            except FileNotFoundError:
                print(f"Warning: Image file not found at {image_full_path}. Skipping.")
                # Add a dummy tensor to maintain list length, or handle this more robustly
                encodings.append(torch.zeros((1, 1536)).to(device))


        encodings_tensor = torch.cat(encodings, dim=0)
        torch.save(encodings_tensor, "encodings.pt")
        print("Encodings saved to encodings.pt")


    # Train-test split
    indices = np.arange(len(train_df))
    train_indices, test_indices = train_test_split(indices, test_size=0.05, random_state=42, stratify=data_df.iloc[:len(train_df)]['answer_type'])
    
    val_indices = np.arange(len(train_df), len(data_df))

    embedding_size = 768
    num_classes = len(np.unique(ans_lb.classes_))
    num_aux_classes = len(np.unique(ans_type_lb.classes_))
    BATCH_SIZE = 64

    train_answers = data_df.iloc[train_indices]['answer'].values
    train_types = data_df.iloc[train_indices]['answer_type'].values
    
    val_answers = data_df.iloc[val_indices]['answer'].values
    val_types = data_df.iloc[val_indices]['answer_type'].values

    train_dataset = VizWizDataset(train_indices, train_answers, train_types, len(train_indices), encodings_tensor)
    val_dataset = VizWizDataset(val_indices, val_answers, val_types, len(val_indices), encodings_tensor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = VQAModel(embedding_size, num_classes, num_aux_classes).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    epochs = 125
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, threshold=1e-6)

    training_loss = []
    validation_loss = []
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = 15

    for e in range(epochs):
        print(f'Epoch: {e+1} | LR: {optimizer.param_groups[0]["lr"]}')
        tr_loss, vl_loss = run_model(model, train_dataloader, val_dataloader, optimizer, device)
        training_loss.append(tr_loss)
        validation_loss.append(vl_loss)
        
        scheduler.step(vl_loss)
        
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved new best model.")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nValidation loss hasn't improved for {early_stopping_patience} epochs. Early stopping.")
            break

    plot_loss(training_loss, validation_loss)
    print("Training finished.")

if __name__ == "__main__":
    main()