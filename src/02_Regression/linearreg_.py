#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import linalg

# 特徴量ベクトルが多次元の場合の線形回帰
class LinearRegression:
    def __init__(self):
        self.w_ = None

    def fit(self, X, t):
        # shape[0]で行数を取得し、n行1列の1行列を生成
        # c_でXと横連結し、~Xに相当する行列を生成
        Xtil = np.c_[np.ones(X.shape[0]), X]
        # ~X^T ~X
        A = np.dot(Xtil.T, Xtil)
        # ~X^T y
        b = np.dot(Xtil.T, t)
        # 1次関数を解く関数 bはn行1列なので1次関数とみなせる？
        self.w_ = linalg.solve(A, b)

    def predict(self, X):
        # 配列の次元(Pythonにおける)数
        # ベクトルの時？
        if X.ndim == 1:
            # 1行にする(=1行ベクトルにする？)
            X = X.reshape(1, -1)
        Xtil = np.c_[np.ones(X.shape[0]), X]
        # ~X^T w
        return np.dot(Xtil, self.w_)


# In[ ]:




