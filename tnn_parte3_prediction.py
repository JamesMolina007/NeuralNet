import torch
from sys import argv
import SimpleLayerNet as SLN
import MultiLayerNet as MLN
from CustomDataset_parte3 import CustomDataset as Dataset
import matplotlib.pyplot as plt

def plotMatrix(className, tp, fp, fn, tn):
    plt.figure()
    plt.title(className)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.imshow([[tp, fp], [fn, tn]], cmap='Blues', interpolation='nearest')
    plt.colorbar()
    # show values on each cell
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format([[tp, fp], [fn, tn]][i][j], 'd'), ha="center", va="center", color="black")
    plt.show()

def main():
    if len(argv) != 3:
        print("Usage: python tnn_parte3_prediction.py <modelo> <archivo_de_datos>")
        exit(1)
    model_file = argv[1]
    data = argv[2]

    tl = torch.load(model_file)
    neurons = len(tl['fc1.weight'])
    model = None
    try:
        tl['fc4.weight']
        #model = MLN.Net(neurons)
    except:
        try:
            tl['fc3.weight']
            model = MLN.Net(neurons)
        except:
            model = SLN.Net(neurons)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    dataset = Dataset(data)
    # mendys burger queen, rigos, wac ronalds, 
    tp_Mendys = 0
    fp_Mendys = 0
    fn_Mendys = 0
    tn_Mendys = 0
    tp_BurgerQueen = 0
    fp_BurgerQueen = 0
    fn_BurgerQueen = 0
    tn_BurgerQueen = 0
    tp_Rigos = 0
    fp_Rigos = 0
    fn_Rigos = 0
    tn_Rigos = 0
    tp_WacRonalds = 0
    fp_WacRonalds = 0
    fn_WacRonalds = 0
    tn_WacRonalds = 0
    
    print("Neural Network output:")
    for i in range(len(dataset)):
        x, y = dataset[i]
        out = model.forward(x)
        list_out = out.tolist()
        cadena = "Mendys" if y == 0 else "Burger Queen" if y == 1 else "Rigos" if y == 2 else "WAC Ronalds"
        prediction = list_out.index(max(list_out))
        prediction = "Mendys" if prediction == 0 else "Burger Queen" if prediction == 1 else "Rigos" if prediction == 2 else "WAC Ronalds"
        if cadena == "Mendys":
            tn_Rigos += 1
            tn_BurgerQueen += 1
            tn_WacRonalds += 1
            if prediction == "Mendys":
                tp_Mendys += 1
            else:
                fn_Mendys += 1
            if prediction == "Burger Queen":
                fp_BurgerQueen += 1
                tn_BurgerQueen -= 1
            elif prediction == "Rigos":
                fp_Rigos += 1
                tn_Rigos -= 1
            elif prediction == "WAC Ronalds":
                fp_WacRonalds += 1
                tn_WacRonalds -= 1
        elif cadena == "Burger Queen":
            tn_Mendys += 1
            tn_Rigos += 1
            tn_WacRonalds += 1
            if prediction == "Burger Queen":
                tp_BurgerQueen += 1
            else:
                fn_BurgerQueen += 1
            if prediction == "Mendys":
                fp_Mendys += 1
                tn_Mendys -= 1
            elif prediction == "Rigos":
                fp_Rigos += 1
                tn_Rigos -= 1
            elif prediction == "WAC Ronalds":
                fp_WacRonalds += 1
                tn_WacRonalds -= 1
        elif cadena == "Rigos":
            tn_Mendys += 1
            tn_BurgerQueen += 1
            tn_WacRonalds += 1
            if prediction == "Rigos":
                tp_Rigos += 1
            else:
                fn_Rigos += 1
            if prediction == "Mendys":
                fp_Mendys += 1
                tn_Mendys -= 1
            elif prediction == "Burger Queen":
                fp_BurgerQueen += 1
                tn_BurgerQueen -= 1
            elif prediction == "WAC Ronalds":
                fp_WacRonalds += 1
                tn_WacRonalds -= 1
        elif cadena == "WAC Ronalds":
            tn_Mendys += 1
            tn_BurgerQueen += 1
            tn_Rigos += 1
            if prediction == "WAC Ronalds":
                tp_WacRonalds += 1
            else:
                fn_WacRonalds += 1
            if prediction == "Mendys":
                fp_Mendys += 1
                tn_Mendys -= 1
            elif prediction == "Burger Queen":
                fp_BurgerQueen += 1
                tn_BurgerQueen -= 1
            elif prediction == "Rigos":
                fp_Rigos += 1
                tn_Rigos -= 1
    print("Mendys")
    print("Accuracy: ", (tp_Mendys + tn_Mendys) / (tp_Mendys + tn_Mendys + fp_Mendys + fn_Mendys))
    precision = tp_Mendys / (tp_Mendys + fp_Mendys)
    recall = tp_Mendys / (tp_Mendys + fn_Mendys)
    print("F1: ", 2 * (precision * recall) / (precision + recall))
    plotMatrix("Mendys", tp_Mendys, fp_Mendys, fn_Mendys, tn_Mendys)
    print("Burger Queen")
    print("Accuracy: ", (tp_BurgerQueen + tn_BurgerQueen) / (tp_BurgerQueen + tn_BurgerQueen + fp_BurgerQueen + fn_BurgerQueen))
    precision = tp_BurgerQueen / (tp_BurgerQueen + fp_BurgerQueen)
    recall = tp_BurgerQueen / (tp_BurgerQueen + fn_BurgerQueen)
    plotMatrix("Burger Queen", tp_BurgerQueen, fp_BurgerQueen, fn_BurgerQueen, tn_BurgerQueen)
    print("F1: ", 2 * (precision * recall) / (precision + recall))
    print("Rigos")
    print("Accuracy: ", (tp_Rigos + tn_Rigos) / (tp_Rigos + tn_Rigos + fp_Rigos + fn_Rigos))
    precision = tp_Rigos / (tp_Rigos + fp_Rigos)
    recall = tp_Rigos / (tp_Rigos + fn_Rigos)
    print("F1: ", 2 * (precision * recall) / (precision + recall))
    plotMatrix("Rigos", tp_Rigos, fp_Rigos, fn_Rigos, tn_Rigos)
    print("WAC Ronalds")
    print("Accuracy: ", (tp_WacRonalds + tn_WacRonalds) / (tp_WacRonalds + tn_WacRonalds + fp_WacRonalds + fn_WacRonalds))
    precision = tp_WacRonalds / (tp_WacRonalds + fp_WacRonalds)
    recall = tp_WacRonalds / (tp_WacRonalds + fn_WacRonalds)
    print("F1: ", 2 * (precision * recall) / (precision + recall))
    plotMatrix("WAC Ronalds", tp_WacRonalds, fp_WacRonalds, fn_WacRonalds, tn_WacRonalds)
    #confusion matrix for mendys using matplotlib
    
    
    




if __name__ == '__main__':
    main()