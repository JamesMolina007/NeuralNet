import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, neurons = 8):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(15,  neurons),
            nn.BatchNorm1d(neurons),
            nn.Sigmoid(),
            nn.Linear(neurons,  neurons*2),
            nn.BatchNorm1d(neurons*2),
            nn.Sigmoid(),
            nn.Linear(neurons*2,  neurons),
            nn.BatchNorm1d(neurons),
            nn.Sigmoid(),
            nn.Linear(neurons,  4),
            nn.LogSoftmax()
        )
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
        self.layers.apply(init_weights)

    def save_state(self, name):
        torch.save(self.state_dict(), "./Classification_Models/" + name + ".pkl")

    def forward(self, x):
        return self.layers(x)
