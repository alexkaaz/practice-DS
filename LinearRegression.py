import matplotlib.pyplot as plt
import pandas as pd


class linearRegressionHM:
    # Определяем глобальные переменные весов
    def __init__(self):
        self.W0 = 0
        self.W1 = 0
        self.DataLen = 0

    # Вычисляет градиент
    def gradient(self, X, y, W0, W1, lr):
        gradient_w0 = 0
        gradient_w1 = 0
        for i in X.index:
            X_el = X[i]
            y_el = y[i]
            gradient_w0 += y_el - ((W1 * X_el) + W0)
            gradient_w1 += X_el * (y_el - ((W1 * X_el) + W0))
        gradient_w0 = -(2/self.DataLen) * gradient_w0
        gradient_w1 = -(2/self.DataLen) * gradient_w1
        W0 = W0 - (lr * gradient_w0)
        W1 = W1 - (lr * gradient_w1)
        return W0, W1

    # Обучает модель
    def fit(self, X, y, lr: float, n: int):
        self.DataLen = len(X)
        fit_w0 = 0
        fit_w1 = 0
        for i in range(n):
            fit_w0, fit_w1 = self.gradient(X, y, fit_w0, fit_w1, lr)
        self.W0 = fit_w0
        self.W1 = fit_w1

    # Предсказывает значение y на основе X использую рассчитаные выше веса w0 и w1
    def predict(self, X_test):
        y_predict = []
        for i in X_test.index:
            X_el = X_test[i]
            y_predict.append((X_el * self.W1) + self.W0)
        return pd.Series(y_predict)

    # Рассчитывает среднюю ошибку для проверки точности
    def MSE(self, y, y_pred):
        dl = len(y)
        sum_of_error = 0
        for i in range(len(y)):
            sum_of_error += (y.iloc[i] - y_pred[i])
        return (1/dl) * sum_of_error

    # Выводит данные на экран
    def chart(self, X, y, y_pred):
        plt.scatter(X, y, color="blue", label="Data Points")
        plt.plot(X, y_pred, color="red", label="Regression Line")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

# Тренировочные данные
df_train = pd.read_csv("data/linear/train.csv")
df_train = df_train.dropna()
X_train = df_train.x
y_train = df_train.y

# Тестовые данные
df_test = pd.read_csv("data/linear/test.csv")
df_test = df_test.dropna()
X_test = df_test.x
y_test = df_test.y

model = linearRegressionHM()
model.fit(X_train, y_train, 0.0001, 1000)
y_pred = model.predict(X_test)
print(model.MSE(y_test, y_pred))
model.chart(X_test, y_test, y_pred)
