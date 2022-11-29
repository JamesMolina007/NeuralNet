for ($i = 0; $i -lt 10; $i++) {  
    python .\tnn_parte2_entrenamiento.py ./Dataset/Train/part2_train_data.csv '5000' >> ("./MSE/tnn_parte_2_result_" + $i + ".csv")
}