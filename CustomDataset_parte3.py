import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path):
        self.data = []
    
        data_csv = pd.read_csv(path)
        
        labels_csv = data_csv.iloc[:, -1]
        labels_csv = labels_csv.replace("Mendys",0)
        labels_csv = labels_csv.replace("Burger Queen",1)
        labels_csv = labels_csv.replace("Rigos",2)
        labels_csv = labels_csv.replace("WAC Ronalds",3)
        self.labels = labels_csv.values.tolist()
        features = data_csv.iloc[:, :-1]
        features = features.replace('Si', 1)
        features = features.replace('No', 0)
        
        for i in range(len(features)):
            data_row = features.iloc[i,:].astype(int).values.tolist()
            self.data.append(data_row)

    def __getitem__(self, index):
        return [torch.tensor(self.data[index],dtype=torch.float),torch.tensor(self.labels[index],dtype=torch.long)]

    def __len__(self):
        return len(self.data)

    