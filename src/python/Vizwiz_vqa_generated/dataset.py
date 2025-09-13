import torch
from torch.utils.data import Dataset

class VizWizDataset(Dataset):
    def __init__(self, indices, answers, types, length, encodings):
        self.indices = indices
        self.answers = answers
        self.types = types
        self.length = length
        self.encd = encodings

    def __getitem__(self, index):
        if self.length <= 20523:
            return self.encd[self.indices[index]].float(), torch.tensor(int(self.answers[index])), torch.tensor(int(self.types[index]))
        
        return self.encd[self.indices[index]].float(), torch.tensor(int(self.answers[index % (self.length // 2)])), torch.tensor(int(self.types[index % (self.length // 2)]))

    def __len__(self):
        return self.length