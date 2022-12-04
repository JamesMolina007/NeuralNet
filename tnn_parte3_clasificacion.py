import os
import torch
from sys import argv
import torch.nn as nn
from MultiLayerNet import Net as MLN
from SimpleLayerNet import Net as SLN
from tnn_parte3_metrics import metrics
from torch.utils.data import DataLoader
from CustomDataset_parte3 import CustomDataset as Dataset


def create_optimizer(Net, learning_rate = 0.01):
    optimizer = torch.optim.SGD(Net.parameters(), lr=learning_rate)
    return optimizer

def testing(model, data):
    dataset = Dataset(data)
    # mendys burger queen, rigos, wac ronalds,     
    Metrics = metrics()
    print("Neural Network output:")
    for i in range(len(dataset)):
        x, y = dataset[i]
        out = model.forward(x)
        list_out = out.tolist()
        Metrics.get_metrics(list_out, y)
    Metrics.print_all_metrics()
    Metrics.plot_all_matrix()

def train_model():
    if len(argv) != 6:
        print("Usage: python tnn_parte3_clasificacion.py <datos_ent> <datos_val> <datos_prb> <num_neurons> <1|2|4>")
        exit(1)
    MetricsTrain = metrics()
    MetricsVal = metrics()
    datos_entrenamiento = argv[1]
    datos_validacion = argv[2]
    datos_prueba = argv[3]
    number_neurons = int(argv[4])
    numero_capas = int(argv[5])
    eps = 0.005
    max_epocas_sin_decremento = 5
    learning_rate = 0.01
    Net = MLN(number_neurons) if numero_capas == 2 else SLN(number_neurons)
    optimizer = create_optimizer(Net, learning_rate)    
    dataset = Dataset(datos_entrenamiento)
    validDataset= Dataset(datos_validacion)
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    previous_prom_loss = 0.0
    epocas_sin_decremento = 0
    min_val = 0.0
    min_val_per_epoch = 0.0
    epoch = 0
    
    for epoch in range(1000):
        print((epoch+1), end=",")
        print("train", end=",")
        prom_loss_train = 0
        for (x, y) in train_dataloader:
            optimizer.zero_grad()
            output = Net.forward(x)
            criterion = torch.nn.NLLLoss()
            loss = criterion(output, y)
            prom_loss_train += loss
            loss.backward()
            optimizer.step()
            for i,out in enumerate(output):
                list_out = out.tolist() 
                #print(list_out, " ---> ", y[i])
                MetricsTrain.get_metrics(list_out, y.tolist()[i])
        MetricsTrain.print_all_metrics()
            # train_acc = torch.sum(output == y)
        
        prom_loss_train = prom_loss_train / len(train_dataloader)
        print("{:.4f}".format(prom_loss_train.item()), end=',')
        print()
        with torch.no_grad():
            print((epoch+1), end=",")
            prom_loss_val = 0
            print("val", end=",")
            for i in range(len(validDataset)):
                x, y = validDataset[i]
                output = Net.forward(x)
                list_out = output.tolist()
                MetricsVal.get_metrics(list_out, y)
                    
                criterion = torch.nn.NLLLoss()
                loss = criterion(output, y)
                prom_loss_val += loss
                    

                if(loss < min_val_per_epoch or min_val_per_epoch == 0):
                    min_val_per_epoch = loss
            
            MetricsVal.print_all_metrics()
            prom_loss_val = prom_loss_val / len(dataset)
            if (min_val < prom_loss_val or min_val == 0):
                min_val = prom_loss_val

            print("{:.4f}".format(prom_loss_val.item()), end=',')
            print()

            # Max epocassin decremento
            if abs(prom_loss_val - previous_prom_loss) > eps and previous_prom_loss != 0:
                break
            previous_prom_loss = prom_loss_val
            if (prom_loss_val > min_val_per_epoch):
                epocas_sin_decremento += 1
            else:
                epocas_sin_decremento = 0
            if (epocas_sin_decremento == max_epocas_sin_decremento):
                break    
    testing(Net, datos_prueba)
    #Net.save_state("parte_3_L" + numero_capas + "_" + str(len(os.listdir("./Pesos/"))) + "_Epochs_" + epoch + "_Neurons_" + str(number_neurons) + "_LearningRate_" + str(learning_rate))

if __name__ == "__main__":
    train_model()