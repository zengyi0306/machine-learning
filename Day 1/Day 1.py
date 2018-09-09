# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:06:54 2018

@author: Harden Zeng
"""

import numpy as np
import pandas as pd
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, : -1].values  ##iloc表示取数据集中的
##某些行和某些列，逗号前表示行，逗号后表示列，这里表示取所有行，列取除了最后
##一列的所有列，因为列是应变量

Y = dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#   imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#  上述代码使用数组X去“训练”一个Imputer类，然后用该类的对象去处理数组Y中的
#缺失值，缺失值的处理方式是使用X中的均值（axis=0表示按列进行）代替Y中的缺失值。

imputer = imputer.fit(X[ : , 1:3])    # 用数据拟合 fit  X的前两列
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print(X_test)


