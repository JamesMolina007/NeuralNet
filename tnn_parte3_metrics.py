import matplotlib.pyplot as plt

class metrics:
    def __init__(self):
        self.cadena_dict = {
            'Mendys': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'Burger Queen': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'Rigos': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'WAC Ronalds': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        }
    
    def get_metrics(self, list_out, y):
        cadena = "Mendys" if y == 0 else "Burger Queen" if y == 1 else "Rigos" if y == 2 else "WAC Ronalds"
        predictionIndex = list_out.index(max(list_out))

        prediction = "Mendys" if predictionIndex == 0 else "Burger Queen" if predictionIndex == 1 else "Rigos" if predictionIndex == 2 else "WAC Ronalds"
        if cadena == "Mendys":
            self.cadena_dict['Rigos']['tn'] += 1
            self.cadena_dict['Burger Queen']['tn'] += 1
            self.cadena_dict['WAC Ronalds']['tn'] += 1
            if prediction == "Mendys":
                self.cadena_dict['Mendys']['tp'] += 1
            else:
                self.cadena_dict['Mendys']['fn'] += 1
            if prediction == "Burger Queen":
                self.cadena_dict['Burger Queen']['fp'] += 1
                self.cadena_dict['Burger Queen']['tn'] -= 1
            elif prediction == "Rigos":
                self.cadena_dict['Rigos']['fp'] += 1
                self.cadena_dict['Rigos']['tn'] -= 1
            elif prediction == "WAC Ronalds":
                self.cadena_dict['WAC Ronalds']['fp'] += 1
                self.cadena_dict['WAC Ronalds']['tn'] -= 1
        elif cadena == "Burger Queen":
            self.cadena_dict['Mendys']['tn'] += 1
            self.cadena_dict['Rigos']['tn'] += 1
            self.cadena_dict['WAC Ronalds']['tn'] += 1
            if prediction == "Burger Queen":
                self.cadena_dict['Burger Queen']['tp'] += 1
            else:
                self.cadena_dict['Burger Queen']['fn'] += 1
            if prediction == "Mendys":
                self.cadena_dict['Mendys']['fp'] += 1
                self.cadena_dict['Mendys']['tn'] -= 1
            elif prediction == "Rigos":
                self.cadena_dict['Rigos']['fp'] += 1
                self.cadena_dict['Rigos']['tn'] -= 1
            elif prediction == "WAC Ronalds":
                self.cadena_dict['WAC Ronalds']['fp'] += 1
                self.cadena_dict['WAC Ronalds']['tn'] -= 1
        elif cadena == "Rigos":
            self.cadena_dict['Mendys']['tn'] += 1
            self.cadena_dict['Burger Queen']['tn'] += 1
            self.cadena_dict['WAC Ronalds']['tn'] += 1
            if prediction == "Rigos":
                self.cadena_dict['Rigos']['tp'] += 1
            else:
                self.cadena_dict['Rigos']['fn'] += 1
            if prediction == "Mendys":
                self.cadena_dict['Mendys']['fp'] += 1
                self.cadena_dict['Mendys']['tn'] -= 1
            elif prediction == "Burger Queen":
                self.cadena_dict['Burger Queen']['fp'] += 1
                self.cadena_dict['Burger Queen']['tn'] -= 1
            elif prediction == "WAC Ronalds":
                self.cadena_dict['WAC Ronalds']['fp'] += 1
                self.cadena_dict['WAC Ronalds']['tn'] -= 1
        elif cadena == "WAC Ronalds":
            self.cadena_dict['Mendys']['tn'] += 1
            self.cadena_dict['Burger Queen']['tn'] += 1
            self.cadena_dict['Rigos']['tn'] += 1
            if prediction == "WAC Ronalds":
                self.cadena_dict['WAC Ronalds']['tp'] += 1
            else:
                self.cadena_dict['WAC Ronalds']['fn'] += 1
            if prediction == "Mendys":
                self.cadena_dict['Mendys']['fp'] += 1
                self.cadena_dict['Mendys']['tn'] -= 1
            elif prediction == "Burger Queen":
                self.cadena_dict['Burger Queen']['fp'] += 1
                self.cadena_dict['Burger Queen']['tn'] -= 1
            elif prediction == "Rigos":
                self.cadena_dict['Rigos']['fp'] += 1
                self.cadena_dict['Rigos']['tn'] -= 1
        
        
    def print_metrics(self, clase):
        print("{:.4f}".format((self.cadena_dict[clase]['tp'] + self.cadena_dict[clase]['tn']) / (self.cadena_dict[clase]['tp'] + self.cadena_dict[clase]['tn'] + self.cadena_dict[clase]['fp'] + self.cadena_dict[clase]['fn'])), end=",")
        denominator = (self.cadena_dict['Mendys']['tp'] + self.cadena_dict[clase]['fp'])
        precision = self.cadena_dict[clase]['tp'] / denominator if denominator != 0 else 0
        denominator = (self.cadena_dict[clase]['tp'] + self.cadena_dict[clase]['fn'])
        recall = self.cadena_dict[clase]['tp'] / denominator if denominator != 0 else 0
        print("{:.4f}".format( (2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0)), end=",")


    def plotMatrix(self, className, tp, fp, fn, tn):
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
        #plt.show()
        plt.savefig("./Matrix/" + className + ".png")

    def plot_matrix(self, clase):
        self.plotMatrix(clase, self.cadena_dict[clase]['tp'], self.cadena_dict[clase]['fp'], self.cadena_dict[clase]['fn'], self.cadena_dict[clase]['tn'])

    def plot_all_matrix(self):
        for clase in self.cadena_dict:
            self.plot_matrix(clase)
        
    def print_all_metrics(self):
        for clase in self.cadena_dict:
            self.print_metrics(clase)
