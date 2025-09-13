# create_encodings.py

import torch
import clip
from PIL import Image
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
from tqdm import tqdm
import pandas as pd
from utils import dataloader_json

# Load data
train_df = dataloader_json("dataset/Annotations/Annotations/train.json")
val_df = dataloader_json("dataset/Annotations/Annotations/val.json")
data_df = pd.concat((train_df, val_df), axis=0, ignore_index=True)

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-L/14", device=device)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model_gpt = OpenAIGPTModel.from_pretrained('openai-gpt')
model_gpt.to(device)

print(f'Using {device}')

# Generate encodings
encodings = []
for img, question in tqdm(zip(data_df['image'], data_df['question'])):
    if "train" in img:
        image_path = f'dataset/train/train/{img}'
    elif "test" in img:
        image_path = f'dataset/test/test/{img}'
    else:
        image_path = f'dataset/val/val/{img}'

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(question).to(device)

    with torch.no_grad():
        image_encoding = model_clip.encode_image(image)
        text_encoding = model_clip.encode_text(text)

        inputs = tokenizer(question, return_tensors="pt").to(device)
        outputs = model_gpt(**inputs)
        last_hidden_states = outputs.last_hidden_state
        gpt_text = last_hidden_states.mean(axis=1)
        
        fused_text = (gpt_text + text_encoding) / 2
        
        encodings.append(torch.cat([image_encoding, fused_text], dim=-1).cpu())

# Save encodings
torch.save(encodings, "withgpt.pt")

print("Encodings saved to withgpt.pt")