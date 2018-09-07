import numpy as np
from numpy import *
from numpy.linalg import *
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("studentscores.csv")
X = dataset.iloc[ : , : 1].values
Y = dataset.iloc[ : , 1 ].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0)

###运用超平面进行线性回归的计算
###a_0
a_0 = ones((len(X_train), 1))
###系数矩阵
A = hstack((X_train, a_0))
###A^T*A*S = A^T*b  b = Y_train

A_1 = inv(A.T.dot(A))
A_2 = A.T.dot(Y_train)
S = A_1.dot(A_2)

def f_1(x, A=S[0], B= S[1]):
    return A*x + B

Y_1 = f_1(X_train)

###画测试集的图

plt.plot(X_train, Y_1, label = 'regression function')

###预测价格
Y_predict = f_1(X_test)
print(Y_predict)

###画训练点
plt.scatter(X_train, Y_train, color = 'red', label = 'train set')

###画测试点
plt.scatter(X_test, Y_test, color = 'black', label = 'test set')
###画预测点
plt.scatter(X_test, Y_predict, color = 'yellow', label = 'prediction')
plt.legend()
plt.show()
