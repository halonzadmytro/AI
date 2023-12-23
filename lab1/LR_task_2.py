import numpy as np
from sklearn import preprocessing
input_data = np.array([[1.3, -3.9, 6.5], [-4.9, -2.2, 1.3],
                      [2.2, 6.5, -6.1], [-5.4, -1.4, 2.2]])
data_binarized = preprocessing.Binarizer(threshold=1.1).transform(input_data)
data_scaled = preprocessing.scale(input_data)
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')


print('\n Бінаризація: \n', data_binarized)
print('\n ДО: ')
print('Mean =', input_data.mean(axis=0))
print("Виключення середнього =", input_data.std(axis=0))
print("\nПІСЛЯ: ")
print("Mean =", data_scaled.mean(axis=0))
print("Виключення середнього: ", data_scaled.std(axis=0))
print("\nМасштабування:\n", data_scaled_minmax)
print("\nL1 нормалізація:\n", data_normalized_l1)
print("\nL2 нормалізація:\n", data_normalized_l2)
