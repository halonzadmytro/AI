import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Розділення даних на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Створення поліноміальних ознак для ступеня 10
poly_features = PolynomialFeatures(degree=10, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)

# Навчання поліноміальної моделі
poly_reg = linear_model.LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Обчислення помилок на тренувальному та тестовому наборі
train_errors, test_errors = [], []
for m in range(1, len(X_train_poly)):
    poly_reg.fit(X_train_poly[:m], y_train[:m])
    y_train_pred = poly_reg.predict(X_train_poly[:m])
    y_test_pred = poly_reg.predict(X_test_poly)
    train_errors.append(mean_squared_error(y_train[:m], y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# Побудова кривих навчання
plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
plt.plot(np.sqrt(test_errors), "b-", linewidth=3, label="test")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("Training set size", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.show()
