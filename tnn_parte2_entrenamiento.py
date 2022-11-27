import torch
from sys import argv
from NeuralNet import Net as NeuralNet

#def update_weights(Net, learning_rate = 0.01):
#    for f in Net.parameters():
        #f.data.sub_(f.grad.data * learning_rate)

def create_optimizer(Net, learning_rate = 0.01):

    for f in Net.parameters():
        print(f)
    optimizer = torch.optim.SGD(Net.parameters(), lr=learning_rate)
    return optimizer

def read_data(datos_entrenamiento):
    with open(datos_entrenamiento, "r") as f:
        data = f.read().splitlines()    
    #ignore first line
    data = [x.split(",") for x in data]
    data = data[1:]
    data = [[float(x[0]),float(x[1]),float(x[2]),float(x[3])] for x in data]
    return data

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
    data = read_data(datos_entrenamiento)
    Net = NeuralNet(number_neurons)
    #print weights
    for f in Net.parameters():
        print(f)
    optimizer = create_optimizer(Net, learning_rate)

    for epoch in range(max_epocas):
        if epoch % 100 == 0:
            print("Epoch: ", epoch+1)
        for i in data:
            optimizer.zero_grad()
            x1,x2 = i[0],i[1]
            y1,y2 = i[2],i[3]
            #print("x1: ", x1, " x2: ", x2, " y1: ", y1, " y2: ", y2)
            output = Net.forward(torch.tensor([x1,x2], dtype=torch.float))
            #print("output: ", output)
            loss = torch.nn.functional.mse_loss(output, torch.tensor([y1,y2], dtype=torch.float))
            loss.backward()
            #print("loss: ", loss)
            optimizer.step()
            #update_weights(Net, learning_rate)

    Net.save_state(2)

if __name__ == "__main__":
    train_model()