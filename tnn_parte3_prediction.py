import torch
from sys import argv
import SimpleLayerNet as NeuralNet
from CustomDataset_parte3 import CustomDataset as Dataset

def main():
    if len(argv) != 3:
        print("Usage: python tnn_parte3_prediction.py <modelo> <archivo_de_datos>")
        exit(1)
    model_file = argv[1]
    data = argv[2]

    tl = torch.load(model_file)
    neurons = len(tl['fc1.weight'])
    model = NeuralNet.Net(neurons)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    dataset = Dataset(data)
    print("Neural Network output:")
    for i in range(len(dataset)):
        x, y = dataset[i]
        out = model.forward(x)
        list_out = out.tolist()
        cadena = "Mendys" if y == 0 else "Burger Queen" if y == 1 else "Rigos" if y == 2 else "WAC Ronalds"
        prediction = list_out.index(max(list_out))
        prediction = "Mendys" if prediction == 0 else "Burger Queen" if prediction == 1 else "Rigos" if prediction == 2 else "WAC Ronalds"
        print("Valor Real: ", cadena, " Valor Predicho: ", prediction)

if __name__ == '__main__':
    main()