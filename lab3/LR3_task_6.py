import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

train_errors = []
val_errors = []
sample_sizes = []

for m in range(1, len(X_train)):
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train[:m], y_train[:m])

    y_train_predict = regressor.predict(X_train[:m])
    train_errors.append(1 - r2_score(y_train[:m], y_train_predict))

    y_val_predict = regressor.predict(X_val)
    val_errors.append(1 - r2_score(y_val, y_val_predict))

    sample_sizes.append(m)

plt.plot(sample_sizes, train_errors, 'o-',
         color='blue', label='Training error')
plt.plot(sample_sizes, val_errors, 'o-', color='red', label='Validation error')
plt.xlabel('Training set size')
plt.ylabel('Error')
plt.legend(loc='best')
plt.show()
