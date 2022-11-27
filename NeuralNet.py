import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, neurons = 2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2,  neurons, bias = True) 
        self.fc2 = nn.Linear(neurons, 2, bias = True)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def save_state(self, part):
        torch.save(self.state_dict(), "./Pesos/pesos_parte_" + str(part) + ".pkl")

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
