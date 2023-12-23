import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# load data from file
# Розділяємо дані на тренувальний та тестовий набор
input_file = 'data_multivar_nb.txt'

data = np.loadtxt(input_file, delimiter=',')
# print(data)
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Створюємо модель машини опорних векторів
svm_model = SVC(kernel='linear')

# Тренуємо модель на тренувальному наборі даних
svm_model.fit(X_train, y_train)

# Діагностичні метрики класифікації
y_pred = svm_model.predict(X_test)


print(classification_report(y_test, y_pred))
