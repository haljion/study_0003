#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from operator import itemgetter

# SVM カーネル法

# カーネル関数の計算クラス
class RBFkernel:
    def __init__(self, X, sigma):
        self.sigma2 = sigma**2
        self.X = X
        self.values_ = np.empty((X.shape[0], X.shape[0]))
    
    # 引数i, jに対してK(xi,xj)を計算する
    def value(self, i, j):
        return np.exp(-((self.X[i, :] - self.X[j, :])**2).sum() / (2*self.sigma2))

    # 引数Z(行列), Zの各行をzkとする
    # s(集合SをBool型の配列で表したもの。計算したいインデックスをTrue, それ以外はFalseとしたもの。)
    # に対してK(xi,zk) (i ∈ S)を計算する
    def eval(self, Z, s):
        return np.exp(-((self.X[s, np.newaxis ,:] - Z[np.newaxis, :, :])**2).sum(axis=2) / (2*self.sigma2))

    
class SVC:
    def __init__(self, C=1., sigma=1., max_iter=10000):
        self.C = C
        self.sigma = sigma
        self.max_iter = max_iter
        
    
    def fit(self, X, y, selections=None):
        a = np.zeros(X.shape[0]) # aの初期値は0
        ay = 0 # Σ(k=1~n) a(k)y(k)
        kernel = RBFkernel(X, self.sigma)
        indices = np.arange(X.shape[0])
        
        for _ in range(self.max_iter):
            s = a != 0.
            # y(t)∇f(a)t
            ydf = y * (1 - y * np.dot(a[s] * y[s], kernel.eval(X, s)).T)
            # iとjの値をargminとargmaxで求める
            iydf = np.c_[indices, ydf]
            i = int(min(iydf[((a > 0) & (y > 0)) | ((a < self.C) & (y < 0))], key=itemgetter(1))[0])
            j = int(max(iydf[((a > 0) & (y < 0)) | ((a < self.C) & (y > 0))], key=itemgetter(1))[0])
            
            if ydf[i] >= ydf[j]:
                break
            
            # aが更新された際の更新処理
            ay2 = ay - y[i] * a[i] - y[j]*a[j]
            kii = kernel.value(i, i)
            kij = kernel.value(i, j)
            kjj = kernel.value(j, j)
            
            s = a != 0.
            s[i] = False
            s[j] = False
            
            kxi = kernel.eval(X[i, :].reshape(1, -1), s).ravel()
            kxj = kernel.eval(X[j, :].reshape(1, -1), s).ravel()
            
            # 目的関数を最大化する^aiの計算
            ai = ((1 - y[i] * y[j] + y[i] * ((kij - kjj)* ay2 - (a[s] * y[s] * (kxi - kxj)).sum()))
                  / (kii + kjj - 2 * kij))

            # ^ai < 0 または ^aj < 0 の時の処理
            # ^ai > C または ^aj > C の時の考慮が加わっているので注意
            if ai < 0:
                ai = 0
            elif ai > self.C:
                ai = self.C
            
            aj = (-ai * y[i] - ay2) * y[j]
            
            if aj < 0:
                aj = 0
                ai = (-aj * y[j] - ay2) * y[i]
            elif aj > self.C:
                aj = self.C
                ai = (-aj * y[j] - ay2) * y[i]
                
            ay += y[i] * (ai - a[i]) + y[j] * (aj - a[j])
            
            if ai == a[i]:
                break
                
            a[i] = ai
            a[j] = aj
            
        self.a_ = a
        self.y_ = y
        self.kernel_ = kernel
        s = a != 0.
        self.w0_ = (y[s] - np.dot(a[s] * y[s], kernel.eval(X[s], s))).sum() / s.sum()
        
        with open("svm.log", "w") as fp:
            print(a, file=fp)
        
        
    def predict(self, X):
        s = self.a_ != 0.
        return np.sign(self.w0_ + np.dot(self.a_[s] * self.y_[s], self.kernel_.eval(X, s)))

