import os
import torch
from sys import argv
import torch.nn as nn
from MultiLayerNet import Net as MLN
from SimpleLayerNet import Net as SLN
from SpecialLayerNet import Net as CLN
from torch.utils.data import DataLoader
from CustomDataset_parte3 import CustomDataset as Dataset


def create_optimizer(Net, learning_rate = 0.01):
    optimizer = torch.optim.SGD(Net.parameters(), lr=learning_rate)
    return optimizer



def train_model():
    if len(argv) != 7:
        print("Usage: python tnn_parte3_entrenamiento.py <datos_entrenamiento> <max_epocas> <datos_validacion> <eps> <max_epocas_sin_decremento> <1|2|4>")
        exit(1)
    datos_entrenamiento = argv[1]
    max_epocas = int(argv[2])
    datos_validacion = argv[3]
    eps = float(argv[4])
    max_epocas_sin_decremento = int(argv[5])
    learning_rate = 0.01
    number_neurons = 32
    if(len(argv) == 6):
        learning_rate = float(argv[3])
        number_neurons = int(argv[4])
        output_file = argv[5]
    
    numero_capas = int(argv[6])
    Net = MLN(number_neurons) if numero_capas == 2 else SLN(number_neurons) if numero_capas == 1 else CLN(number_neurons)

    optimizer = create_optimizer(Net, learning_rate)    

    dataset = Dataset(datos_entrenamiento)
    validDataset= Dataset(datos_validacion)
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    previous_prom_loss = 0.0
    epocas_sin_decremento = 0
    min_val = 0.0
    min_val_per_epoch = 0.0
    for epoch in range(max_epocas):
        print((epoch+1), end=',')
        prom_loss_train = 0
        for (x, y) in train_dataloader:
            optimizer.zero_grad()
            output = Net.forward(x)
            criterion = torch.nn.NLLLoss()
            loss = criterion(output, y)
            prom_loss_train += loss
            loss.backward()
            optimizer.step()
            # train_acc = torch.sum(output == y)
        
        prom_loss_train = prom_loss_train / len(train_dataloader)
        print(prom_loss_train.item(), end=',')

        with torch.no_grad():
            prom_loss = 0
            for i in range(len(validDataset)):
                x, y = dataset[i]
                output = Net.forward(x)
                criterion = torch.nn.NLLLoss()
                loss = criterion(output, y)
                prom_loss += loss
                if(loss < min_val_per_epoch or min_val_per_epoch == 0):
                    min_val_per_epoch = loss
            
            prom_loss = prom_loss / len(dataset)
            if (min_val < prom_loss or min_val == 0):
                min_val = prom_loss

            print(prom_loss.item(), end=',')
            print()

            # Max epocassin decremento
            if abs(prom_loss - previous_prom_loss) > eps and previous_prom_loss != 0:
                break
            previous_prom_loss = prom_loss
            if (prom_loss > min_val_per_epoch):
                epocas_sin_decremento += 1
            else:
                epocas_sin_decremento = 0
            if (epocas_sin_decremento == max_epocas_sin_decremento):
                break    
    Net.save_state("parte_3_L" + argv[6] + "_" + str(len(os.listdir("./Pesos/"))) + "_Epochs_" + str(max_epocas) + "_Neurons_" + str(number_neurons) + "_LearningRate_" + str(learning_rate))

if __name__ == "__main__":
    train_model()