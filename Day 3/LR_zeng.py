import sys
import pandas as pd
import numpy as np

def fit(X_train, Y_train):
###a_0
    a_0 = np.ones((len(X_train), 1))
###系数矩阵
    A = np.hstack((X_train, a_0))
###A^T*A*S = A^T*b  b = Y_train

    A_1 = np.linalg.inv(A.T.dot(A))
    A_2 = A.T.dot(Y_train)
    coefficient = A_1.dot(A_2)
    return coefficient

def pred(X_test, A):

    return X_test.dot(A[: -1]) + A[-1]


