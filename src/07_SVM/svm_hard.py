#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from operator import itemgetter

# ハードマージンSVM
class SVC:
    def fit(self, X, y, selections=None):
        a = np.zeros(X.shape[0]) # aの初期値は0
        ay = 0 # Σ(k=1~n) a(k)y(k)
        ayx = np.zeros(X.shape[1]) # Σ(k=1~n) a(k)y(k)x(k)
        yx = y.reshape(-1, 1) * X
        indices = np.arange(X.shape[0])
        
        while True:
            # y(t)∇f(a)t
            ydf = y * (1-np.dot(yx, ayx.T))
            # iとjの値をargminとargmaxで求める
            iydf = np.c_[indices, ydf]
            i = int(min(iydf[(y < 0) | (a > 0)], key=itemgetter(1))[0])
            j = int(max(iydf[(y > 0) | (a > 0)], key=itemgetter(1))[0])
            
            if ydf[i] >= ydf[j]:
                break
            
            # aが更新された際の更新処理
            ay2 = ay - y[i] * a[i] - y[j]*a[j]
            ayx2 = ayx - y[i] * a[i] * X[i, :] - y[j] * a[j] * X[j, :]
            # 目的関数を最大化する^aiの計算
            ai = ((1 - y[i] * y[j] + y[i] * np.dot(X[i, :] - X[j, :], X[j, :] * ay2 - ayx2))
                  / ((X[i] - X[j])**2).sum())
            aj = (-ai * y[i] - ay2) * y[j]
            
            # ^ai < 0 または ^aj < 0 の時の処理
            if ai < 0:
                ai = 0
                aj = (-ai * y[i] - ay2) * y[j]
            
            if aj < 0:
                aj = 0
                ai = (-aj * y[j] - ay2) * y[i]
                
            ay += y[i] * (ai - a[i]) + y[j] * (aj - a[j])
            ayx += y[i] * (ai - a[i]) * X[i, :] + y[j] * (aj - a[j]) *X [j, :]
            
            if ai == a[i]:
                break
                
            a[i] = ai
            a[j] = aj
            
        self.a_ = a
        ind = a != 0.
        self.w_ = ((a[ind] * y[ind]).reshape(-1, 1) * X[ind, :]).sum(axis=0)
        self.w0_ = (y[ind] - np.dot(X[ind, :], self.w_)).sum() / ind.sum()
        
        
    def predict(self, X):
        return np.sign(self.w0_ + np.dot(X, self.w_))

