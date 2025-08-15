import pandas as pd
from math import log2

class DecisionTree:

    def __init__(self):
        self.ig_per_class = {}
        self.base_entropy = 0

    def __entropy__(self, y):
        classes = {} 
        entropy = 0 
        for el in y:
            if el not in classes:
                classes[el] = 0
            classes[el] += 1
        for p in classes.values():
            entropy += -(p/len(y)) * log2((p/len(y)))
        return entropy

    def __info_gain__(self, data, data_per_col):
        data_entropy = {}
        n = 0
        for el in data:
            data_entropy[el] = self.__entropy__(data[el])
        for v in data_entropy:
            n += (len(data[v])/data_per_col) * data_entropy[v]
        return self.base_entropy - n
     
    def __col_content__(self, X, y):
        ig_per_class = {}
        for col in X.columns:  
            data_per_col = len(X[col])
            col_cont = {}     
            for i in range(len(X[col])): 
                value = X[col].iloc[i]   
                class_label = y.iloc[i]   
                if value not in col_cont:
                    col_cont[value] = []  
                col_cont[value].append(class_label)  
            counted_cont = {k: v for k, v in col_cont.items()}
            ig_per_class[col] = self.__info_gain__(counted_cont, data_per_col)  
        return ig_per_class

    def fit(self, X, y):
        self.base_entropy = self.__entropy__(y)
        self.ig_per_class = self.__col_content__(X, y)
        print(self.base_entropy)
        print(self.ig_per_class)

model = DecisionTree()
data = pd.read_csv('...')
y = data['...']
X = data[['...', ..., '...']]
model.fit(X, y)
