from sys import argv
import NeuralNet
import torch

def main():
    if len(argv) != 3:
        print("Usage: python tnn_parte2_prediction.py <modelo> <archivo_de_datos>")
        exit(1)
    model_file = argv[1]
    data = argv[2]

    tl = torch.load(model_file)
    neurons = len(tl['fc1.weight'])
    model = NeuralNet.Net(neurons)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    with open(data, "r") as f:
        data = f.read().splitlines()
    data = [x.split(",") for x in data]
    data = [[float(x[0]),float(x[1])] for x in data]
    print("Neural Network output:")
    
    for i in data:
        x1,x2 = i[0],i[1]
        out = model.forward(torch.tensor([x1,x2], dtype=torch.float))
        list_out = out.tolist()
        print(list_out[0],',',list_out[1])
        

if __name__ == '__main__':
    main()