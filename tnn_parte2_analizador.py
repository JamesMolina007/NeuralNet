import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    pandas_list = []
    for subdir, dirs, files in os.walk("./MSE/"):
        for fileName in files:
            if fileName.split('.')[1] == "csv":
                df = pd.read_csv(os.path.join(subdir,fileName), index_col=0, header=None, encoding='utf_16_le')
                #remove last column
                df = df.drop(df.columns[[-1]], axis=1)
                pandas_list.append(df)
                
    epochs = pandas_list[0].shape[0]
    epochs_list = [i for i in range(epochs)]
    max_list = []
    min_list = []
    mean_list = []
    for j in range(epochs):
        max = 0
        min = -1
        mean = 0
        for i in pandas_list:
            for k in i.iloc[j]:
                if k > max:
                    max = k
                if k < min or min == -1:
                    min = k
                mean += k
        mean = mean / (len(pandas_list) * len(pandas_list[0].iloc[j]))
        max_list.append(max)
        min_list.append(min)
        mean_list.append(mean)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("MSE vs Epochs")
    plt.xticks([1, len(epochs_list)])
    plt.plot(epochs_list, max_list, label="Max")
    plt.plot(epochs_list, min_list, label="Min")
    plt.plot(epochs_list, mean_list, label="Mean")
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    main()