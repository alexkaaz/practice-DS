# Примените нейронную сеть к набору данных о занятости сотрудников, с которым
# мы работали в главе 6. Данные можно импортировать отсюда (https://tinyurl.com/
#  y6r7qjrp). Попробуйте построить нейронную сеть так, чтобы она давала прогнозы
# на этом наборе данных. Чтобы оценить ее эффективность, используйте метрику
# точности и матрицы ошибок. Хорошая ли это модель для этой задачи? Почему
# да или почему нет?
#  Хотя нейронную сеть можно построить с нуля, для экономии времени вос
# пользуйтесь scikit-learn, PyTorch или другой библиотекой глубокого обучения.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

Data = pd.read_csv("https://tinyurl.com/y6r7qjrp", delimiter=",")

# L = 0.05

X = Data.values[:, :-1]
Y = Data.values[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)
# n = X_train.shape[0]
#
# w_hidden = np.random.rand(3, 3)
# w_output = np.random.rand(1, 3)
#
# b_hidden = np.random.rand(3, 1)
# b_output = np.random.rand(1, 1)
#
# relu = lambda x: np.maximum(x, 0)
# logistic = lambda x: 1 / (1 + np.exp(-x))
#
# def forward_prop(X):
#     Z1 = w_hidden @ X + b_hidden
#     A1 = relu(Z1)
#     Z2 = w_output @ A1 + b_output
#     A2 = logistic(Z2)
#     return Z1, A1, Z2, A2
#
# d_relu = lambda x: x > 0
# d_logistic = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2
#
# def backword_prop(Z1, A1, Z2, A2, X, Y):
#     dC_dA2 = 2 * A2 - 2 * Y
#     dA2_dZ2 = d_logistic(Z2)
#     dZ2_dA1 = w_output
#     dZ2_dW2 = A1
#     dZ2_dB2 = 1
#     dA1_dZ1 = d_relu(Z1)
#     dZ1_dW1 = X
#     dZ1_dB1 = 1
#
#     dC_dW2 = dC_dA2 @ dA2_dZ2 @ dZ2_dW2.T
#
#     dC_dB2 = dC_dA2 @ dA2_dZ2 * dZ2_dB2
#
#     dC_dA1 = dC_dA2 @ dA2_dZ2 @ dZ2_dA1
#
#     dC_dW1 = dC_dA1 @ dA1_dZ1 @ dZ1_dW1.T
#
#     dC_dB1 = dC_dA1 @ dA1_dZ1 * dZ1_dB1
#
#     return dC_dW1, dC_dB1, dC_dW2, dC_dB2
#
# for i in range(100000):
#     idx = np.random.choice(n, 1, replace=False)
#     X_sample = X_train[idx].transpose()
#     Y_sample = Y_train[idx]
#
#     Z1, A1, Z2, A2 = forward_prop(X_sample)
#
#     dW1, dB1, dW2, dB2 = backword_prop(Z1, A1, Z2, A2, X_sample, Y_sample)
#
#     w_hidden -= L * dW1
#     b_hidden -= L * dB1
#     w_output -= L * dW2
#     b_output -= L * dB2
#
# def predict_proba(A, P, Y):
#     X = np.array([[A, P, Y]]).transpose()
#     Z1, A1, Z2, A2 = forward_prop(X)
#     return A2

# while True:
#     par = input()
#     (A, P, Y) = par.split(",")
#     print(predict_proba(int(A), int(P), int(Y)))

nn = MLPClassifier(solver="sgd",
                   hidden_layer_sizes=(3, ),
                   activation="relu",
                   max_iter=100_000,
                   learning_rate_init=.05)
nn.fit(X_train, Y_train)

print("Матрица ошибок:")
matrix = confusion_matrix(y_true=Y_test, y_pred=nn.predict(X_test))
print(matrix)

print(nn.coefs_)
print(nn.intercepts_)
print(f"Средняя точность на обучающей выборке: {nn.score(X_train, Y_train)}")
print(f"Средняя точность на тестовой выборке: {nn.score(X_test, Y_test)}")
