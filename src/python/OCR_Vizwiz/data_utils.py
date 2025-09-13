import os
import json
from torch.utils.data import Dataset
from PIL import Image

class VizWizOCRDataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(os.path.join(data_path, "annotations.json"), "r") as f:
            self.annotations = json.load(f)
        self.image_path = os.path.join(data_path, "images")
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        image = Image.open(os.path.join(self.image_path, sample["image"])).convert("RGB")
        question = sample["question"]
        answer = sample.get("answer", None)

        if self.transform:
            image = self.transform(image)

        return image, question, answer
