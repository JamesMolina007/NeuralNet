
import torch
from NeuralNet import Net as NeuralNet
from sys import argv

def feed_forward(_net,x):
    out = _net.forward(x)
    list_out = out.tolist()
    print(list_out[0],',',list_out[1])

def read_data(datos_entrenamiento):
    with open(datos_entrenamiento, "r") as f:
        data = f.read().splitlines()    
    data = [x.split(",") for x in data]
    data = [[float(x[0]),float(x[1])] for x in data]
    return data

def printWeighs(model):
    for param in model.parameters():
        print(param.data.tolist())


def main():
    if len(argv) != 2:
        print("Usage: python tnn_parte1_feedforward.py <archivo_de_pesos>")
        exit(1)
    
    _net = NeuralNet()
    print("Pesos aleatorios de la red: ")
    printWeighs(_net)
    
    _net.save_state("pesos_parte_1")

    archivo_de_pesos = argv[1]
    
    data = read_data(archivo_de_pesos)

    print("Neural Network output:")
    for i in data:
        x1,x2 = i[0],i[1]
        feed_forward(_net,torch.tensor([x1,x2], dtype=torch.float))


if __name__ == '__main__':
    main()