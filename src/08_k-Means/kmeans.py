#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import itertools

# k-Means法
class KMeans:
    def __init__(self, n_clusters, max_iter=1000, random_seed=0):
        self.n_clusters = n_clusters # クラスタ数
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)
    
    
    def fit(self, X):
        # 0からn_clusters-1 の値を繰り返すiterableなオブジェクト
        cycle = itertools.cycle(range(self.n_clusters))
        # 初期値として乱数でラベルを決める
        self.labels_ = np.fromiter(
            itertools.islice(cycle, X.shape[0]), dtype=np.int
        )
        self.random_state.shuffle(self.labels_)
        labels_prev = np.zeros(X.shape[0])
        count = 0
        self.cluster_centers_ = np.zeros(
            (self.n_clusters, X.shape[1])
        )
        # 各点のクラスタへの所属情報が変化しない状態、もしくは最大繰り返し回数に到達するまでループ
        while(not (self.labels_ == labels_prev).all() and count < self.max_iter):
            for i in range(self.n_clusters):
                # クラスタの重心の計算
                XX = X[self.labels_ == i, :]
                self.cluster_centers_[i, :] = XX.mean(axis=0)
            
            # 距離の計算
            dist = ((X[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :])**2).sum(axis=1)
            # 対応ラベルの計算
            labels_prev = self.labels_
            self.labels_ = dist.argmin(axis=1)
            
            count += 1
    
    
    def predict(self, X):
        dist = ((X[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :])**2).sum(axis=1)
        labels = dist.argmin(axis=1)
        return labels

