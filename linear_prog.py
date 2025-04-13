from random import shuffle

import pandas as pd

from sklearn.linear_model import LinearRegression
from scipy.stats import t
from math import sqrt
from sklearn.model_selection import cross_val_score, ShuffleSplit


df = pd.read_csv("https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/regression/linear_normal.csv", delimiter=",")

X = df.values[:, :-1]

Y = df.values[:, -1]

fit = LinearRegression().fit(X, Y)

m = fit.coef_.flatten()
b = fit.intercept_.flatten()
print(f"m = {m} and b = {b}")

correlations = df.corr(method="pearson")
print(correlations)

r = 0.92421
n = len(df)
lower_cv = t(n-1).ppf(.025)
upper_cv = t(n-1).ppf(.925)

test_value = r / sqrt((1-r**2) / (n-2))

if test_value < lower_cv or test_value > upper_cv:
    print("Корреляция обоснована, отвергаем H₀")
else:
    print("Корреляция не обоснована, нельзя отвергнуть H₀")

if test_value > 0:
    p_value = 1.0 - t(n-1).cdf(test_value)
else:
    p_value = t(n-1).cdf(test_value)

print(f"p-value = {p_value}")

points =  list(pd.read_csv("https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/regression/linear_normal.csv", delimiter=",").itertuples())

pn = len(points)


x_0 = 50
x_mean = sum(p.x for p in points) / len(points)


t_value = t(pn-2).ppf(.975)

standard_error = sqrt(sum((p.y - (m[0] * p.x + b[0])) ** 2 for p in points) / (pn - 2))

margin_of_error = t_value * standard_error * \
                  sqrt(1 + (1/pn) + (pn * (x_0 - x_mean) ** 2) / \
                  (pn * sum(p.x ** 2 for p in points) - \
                  sum(p.x for p in points) ** 2))

predicted_y = m[0]*x_0 +b[0]

print(predicted_y - margin_of_error, predicted_y + margin_of_error)


kfold = ShuffleSplit(n_splits=3, test_size=.33, random_state=7)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(results)
print("MSE: mean = %.3f (stdev = %.3f)" % (results.mean(), results.std()))

#  4. Начните регрессию заново и разделите данные на обучающую и тестовую
# выборки. Не стесняйтесь экспериментировать с перекрестной и случайной
# валидацией. Насколько хорошо и стабильно работает линейная регрессия
# на тестовых данных? Почему так