import os
import torch
from sys import argv
import torch.nn as nn
from NeuralNet import Net as NeuralNet
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset as Dataset

#def update_weights(Net, learning_rate = 0.01):
#    for f in Net.parameters():
        #f.data.sub_(f.grad.data * learning_rate)

def create_optimizer(Net, learning_rate = 0.01):
    optimizer = torch.optim.SGD(Net.parameters(), lr=learning_rate)
    return optimizer

def train_model():
    if len(argv) != 3:
        print("Usage: python tnn_parte2_entrenamiento.py <datos_entrenamiento> <max_epocas>")
        exit(1)
    datos_entrenamiento = argv[1]
    max_epocas = int(argv[2])
    learning_rate = 0.01
    number_neurons = 32
    if(len(argv) == 6):
        learning_rate = float(argv[3])
        number_neurons = int(argv[4])
        output_file = argv[5]
    
    Net = NeuralNet(number_neurons)
    optimizer = create_optimizer(Net, learning_rate)
    
    dataset = Dataset(datos_entrenamiento)
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for epoch in range(max_epocas):
        print((epoch+1), end=',')
        for (x, y) in train_dataloader:
            optimizer.zero_grad()
            output = Net.forward(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prom_loss = 0
            for i in range(len(dataset)):
                x, y = dataset[i]
                output = Net.forward(x)
                prom_loss += torch.nn.functional.mse_loss(output, y)
            prom_loss = prom_loss / len(dataset)
            print(prom_loss.item(), end=',')
            print()
    #####    
        
    Net.save_state("pesos_modelo_" + str(len(os.listdir("./Pesos/"))) + "_Epochs_" + str(max_epocas) + "_Neurons_" + str(number_neurons) + "_LearningRate_" + str(learning_rate))

if __name__ == "__main__":
    train_model()