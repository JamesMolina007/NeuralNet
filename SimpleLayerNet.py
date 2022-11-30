import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, neurons = 8):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(15,  neurons, bias = True) 
        self.fc2 = nn.Linear(neurons, 4, bias = True)
        self.m = nn.LogSoftmax(dim=0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def save_state(self, name):
        torch.save(self.state_dict(), "./Classification_Models/" + name + ".pkl")

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.nn.functional.log_softmax(self.m(x),-1)
        return x
