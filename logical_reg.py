import pandas as pd
from scipy.stats import probplot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix


df = pd.read_csv("https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/classification/light_dark_font_training_set.csv",
                 delimiter=",")
X = df.values[:, :-1]
Y = df.values[:, -1]

kfold = KFold(n_splits=3, shuffle=True)
fit = LogisticRegression(penalty=None)
res = cross_val_score(fit, X, Y, cv=kfold)

print(f"Точность = {res.mean():.3f}, стандартное отклонение = {res.std():.3f}")

model = LogisticRegression(solver="liblinear")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33,
                                                    random_state=10)
model.fit(X_train, Y_train)
prediction = model.predict(X_test)

matrix = confusion_matrix(y_true=Y_test, y_pred=prediction)
print(matrix)

data = pd.read_csv("https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/classification/light_dark_font_training_set.csv")
inputs = data.iloc[:, :-1].to_numpy()
output = data.iloc[:, -1].to_numpy()
font_col = LogisticRegression(penalty=None).fit(inputs, output)

def preditc_font_color(R, G, B):
    prediction = font_col.predict([[R, G, B]])
    return prediction

# while True:
#     n = input("R, G, B: ")
#     R, G, B = n.split(",")
#     print(preditc_font_color(int(R), int(G), int(B)))
# По ссылке (https://bit.ly/3imidqa) представлен набор данных с тремя входными
# переменными RED, GREEN и BLUE, которые задают цвет фона в формате RGB,
# а также выходной переменной LIGHT_OR_DARK_FONT_IND, которая прогнозирует,
# какой шрифт лучше подойдет для этого фона — светлый (0) или темный (1).


#  4. Сделайте вывод на основе предыдущих упражнений: эффективна ли логи
# стическая регрессия для того, чтобы прогнозировать тип шрифта для за
# данного цвета фона?