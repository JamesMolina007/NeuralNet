import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, "r") as f:
            self.data = f.read().splitlines()
        self.data = self.data[1:]
        self.data = [x.split(",") for x in self.data]
        self.data = [[int(x[0]),int(x[1]),int(x[2]),int(x[3])] for x in self.data]

    def __getitem__(self, index):
        return [torch.tensor([self.data[index][0], self.data[index][1]], dtype=torch.float), \
            torch.tensor([self.data[index][2], self.data[index][3]], dtype=torch.float),]

    def __len__(self):
        return len(self.data)

    