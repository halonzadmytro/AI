import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Перетворення вхідних даних
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Модель регресії
reg = linear_model.LinearRegression()
reg.fit(X_train_poly, y_train)

# Побудова графіку
X_plot = np.linspace(-4, 2, 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = reg.predict(X_plot_poly)

plt.scatter(X_train, y_train, s=10)
plt.scatter(X_test, y_test, s=10)
plt.plot(X_plot, y_plot, color='r')
plt.show()

# Оцінка регресії
y_train_pred = reg.predict(X_train_poly)
y_test_pred = reg.predict(X_test_poly)

print('Train R2 score:', r2_score(y_train, y_train_pred))
print('Test R2 score:', r2_score(y_test, y_test_pred))
