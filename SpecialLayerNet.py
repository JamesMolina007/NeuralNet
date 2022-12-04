import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, neurons = 8):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(15,  neurons) 
        self.fc2 = nn.Linear(neurons,  neurons) 
        self.fc3 = nn.Linear(neurons,  neurons) 
        self.fc4 = nn.Linear(neurons, 4)
        self.m = nn.LogSoftmax()
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)

    def save_state(self, name):
        torch.save(self.state_dict(), "./Classification_Models/" + name + ".pkl")

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.m(self.fc4(x))
        return x
