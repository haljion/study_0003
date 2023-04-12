#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import kmeans

# 乱数によるデータ作成
np.random.seed(0)
points1 = np.random.randn(50, 2)
# (0, 5)を中心とした乱数(全ての点が(0,5)方向に平行移動する)
points2 = np.random.randn(50, 2) + np.array([5, 0])
# (5, 5)を中心とした乱数
points3 = np.random.randn(50, 2) + np.array([5, 5])

points = np.r_[points1, points2, points3]
np.random.shuffle(points)

# 作成したモデルのインスタンス
model = kmeans.KMeans(3)
model.fit(points)

markers = ["+", "*", "o"]

for i in range(3):
    p = points[model.labels_ == i, :]
    # 点群の描画
    plt.scatter(p[:, 0], p[:, 1], color="k", marker=markers[i])

plt.show()


# In[ ]:




