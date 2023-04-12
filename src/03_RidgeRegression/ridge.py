#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy import linalg

# リッジ回帰
class RidgeRegression:
    
    def __init__(self, lambda_=1.):
        self.lambda_ = lambda_
        self.w_ = None

    def fit(self, X, t):
        # shape[0]で行数を取得し、n行1列の1行列を生成
        # c_でXと横連結し、~Xに相当する行列を生成
        Xtil = np.c_[np.ones(X.shape[0]), X]
        # c: [~Xの列数]*[~Xの列数]の単位行列(I)
        c = np.eye(Xtil.shape[1])
        # ~X^T ~X
        A = np.dot(Xtil.T, Xtil) + self.lambda_ * c
        # ~X^T y
        b = np.dot(Xtil.T, t)
        # 1次関数を解く関数 bはn行1列なので1次関数とみなせる？
        self.w_ = linalg.solve(A, b)

    def predict(self, X):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        # ~X^T w
        return np.dot(Xtil, self.w_)

