#python tnn_parte3_clasificacion.py ./Dataset/Train/training_data_very_large.csv ./Dataset/Validation/validation_data.csv ./Dataset/Testing/testing_data.csv 8 1 >> ./NLLLOSS/tnn_parte3_N8_l1_result.csv
#python tnn_parte3_analizador.py  ./NLLLOSS/tnn_parte3_N8_l1_result.csv ./Plots/1-Capa-8-Neu

python tnn_parte3_clasificacion.py ./Dataset/Train/training_data_very_large.csv ./Dataset/Validation/validation_data.csv ./Dataset/Testing/testing_data.csv 8 2 >> ./NLLLOSS/tnn_parte3_N8_l2_result.csv
python tnn_parte3_analizador.py  ./NLLLOSS/tnn_parte3_N8_l1_result.csv ./Plots/2-Capa-8-Neu

python tnn_parte3_clasificacion.py ./Dataset/Train/training_data_very_large.csv ./Dataset/Validation/validation_data.csv ./Dataset/Testing/testing_data.csv 16 1 >> ./NLLLOSS/tnn_parte3_N16_l1_result.csv
python tnn_parte3_analizador.py  ./NLLLOSS/tnn_parte3_N8_l1_result.csv ./Plots/1-Capa-16-Neu

#python tnn_parte3_clasificacion.py ./Dataset/Train/training_data_very_large.csv ./Dataset/Validation/validation_data.csv ./Dataset/Testing/testing_data.csv 16 2 >> ./NLLLOSS/tnn_parte3_N16_l2_result.csv
#python tnn_parte3_analizador.py  ./NLLLOSS/tnn_parte3_N8_l1_result.csv ./Plots/2-Capa-16-Neu

#python tnn_parte3_clasificacion.py ./Dataset/Train/training_data_very_large.csv ./Dataset/Validation/validation_data.csv ./Dataset/Testing/testing_data.csv 16 4 >> ./NLLLOSS/tnn_parte3_custom_result.csv
#python tnn_parte3_analizador.py  ./NLLLOSS/tnn_parte3_custom_result.csv ./Plots/Red_Custom