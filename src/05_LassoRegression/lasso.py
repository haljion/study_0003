#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Lasso回帰のアルゴリズム
import numpy as np

# ソフト閾値関数
def soft_thresholding(x, y):
    return np.sign(x) * max(abs(x) - y, 0)


class Lasso:
    def __init__(self, lambda_, tol=0.0001, max_iter=1000):
        # λ
        self.lambda_ = lambda_
        # 収束判定のためのトレランス(許容度)
        self.tol = tol
        # 最大繰り返し数
        self.max_iter = max_iter
        self.w_ = None
    
    
    def fit(self, X, t):
        # 行数、列数
        n, d = X.shape
        self.w_ = np.zeros(d + 1)
        avgl1 = 0.
        # 収束条件が満たされなくても、max_iter回繰り返したときは終了
        for _ in range(self.max_iter):
            avgl1_prev = avgl1
            self._update(n, d, X, t)
            avgl1 = np.abs(self.w_).sum() / self.w_.shape[0]
            # 収束条件の例
            # wのL1ノルムを次元数で割ったもの(|w|/d)の変化量がtol以下になるまで繰り返す
            if abs(avgl1 - avgl1_prev) <= self.tol:
                break


    def _update(self, n, d, X, t):
        self.w_[0] = (t - np.dot(X, self.w_[1:])).sum() / n
        # w0
        w0vec = np.ones(n) * self.w_[0]
        
        for k in range(d):
            ww = self.w_[1:]
            ww[k] = 0
            # Σ{y(i) - w0 - Σ(j!=k)x(ij)w(j)}
            q = np.dot(t - w0vec - np.dot(X, ww), X[:, k])
            # Σx(ij)^2
            r = np.dot(X[:, k], X[:, k])
            self.w_[k + 1] = soft_thresholding(q / r, self.lambda_)
            
            
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xtil = np.c_[np.ones(X.shape[0]), X]
        # ~X^T w
        return np.dot(Xtil, self.w_)

