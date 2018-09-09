import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import sys
sys.path.append(r'C:\Users\Harden Zeng\Desktop\machine-learning\Day 3')

import LR_zeng



dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
#regressor = LR_zeng

A = regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)
#y_pred = regressor.pred(X_test, A)
print(y_pred)

