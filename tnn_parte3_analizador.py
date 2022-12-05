import os
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt



def save_plot(className, epochs_list, val1, lbl1, lbl2, val2, train = False, hayVal = False):
    output_folder = argv[2]
    plt.clf()
    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.title(className + "\n" + ("Train" if train else "Validation") + " Performance vs Epochs")
    plt.xticks([1, len(epochs_list)])
    plt.plot(epochs_list, val1, label=lbl1)
    if hayVal:
        plt.plot(epochs_list, val2, label=lbl2)
    plt.legend()
    type = "_train" if train else "_validation"
    metrics = "F1Acc_" if hayVal else "NLLLOSS_"
    plt.savefig(output_folder + "/" + metrics + className + type + ".png")

def main():

    if len(argv) != 3:
        print("Usage: python tnn_parte3_analizador.py <archivo_de_resultados> <output_folder>")
        exit(1)
    data_name = argv[1]
    df = pd.read_csv(data_name, index_col=0, header=None, encoding='utf_16_le')
    df = df.drop(df.columns[[-1]], axis=1)
    
    epochs = int((df.shape[0] - 2)/2)
    epochs_list = [i for i in range(epochs)]
    
    #get rows with value validation
    df_val = df[df[1] == 'val']
    df_train = df[df[1] == 'train']
    
    save_plot("Mendys", epochs_list, df_train[:][2], 'F1-Score', 'Accuracy', df_train[:][3], train = True, hayVal = True)
    save_plot("Burger Queen", epochs_list, df_train[:][4], 'F1-Score', 'Accuracy', df_train[:][5], train = True, hayVal = True)
    save_plot("Rigos", epochs_list, df_train[:][6], 'F1-Score', 'Accuracy', df_train[:][7], train = True, hayVal = True)
    save_plot("Wac Ronalds", epochs_list, df_train[:][8], 'F1-Score', 'Accuracy', df_train[:][9], train = True, hayVal = True)
        
    save_plot("Mendys", epochs_list, df_val[:][2], 'F1-Score', 'Accuracy', df_val[:][3], hayVal = True)
    save_plot("Burger Queen", epochs_list, df_val[:][4], 'F1-Score', 'Accuracy', df_val[:][5], hayVal = True)
    save_plot("Rigos", epochs_list, df_val[:][6], 'F1-Score', 'Accuracy', df_val[:][7], hayVal = True)
    save_plot("Wac Ronalds", epochs_list, df_val[:][8], 'F1-Score', 'Accuracy', df_val[:][9], hayVal = True)
    
    save_plot("Mendys", epochs_list, df_train[:][10], 'NLLLOSS', 0, 0, train = True)
    save_plot("Burger Queen", epochs_list, df_train[:][10], 'NLLLOSS',0,0, train = True)
    save_plot("Rigos", epochs_list, df_train[:][10], 'NLLLOSS',0,0, train = True)
    save_plot("Wac Ronalds", epochs_list, df_train[:][10], 'NLLLOSS',0,0, train = True)

    save_plot("Mendys", epochs_list, df_val[:][10], 'NLLLOSS',0,0)
    save_plot("Burger Queen", epochs_list, df_val[:][10], 'NLLLOSS',0,0)
    save_plot("Rigos", epochs_list, df_val[:][10], 'NLLLOSS',0,0)
    save_plot("Wac Ronalds", epochs_list, df_val[:][10], 'NLLLOSS',0,0)


if __name__ == '__main__':
    main()
