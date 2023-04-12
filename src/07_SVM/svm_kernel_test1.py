#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import svm

plt.axes().set_aspect("equal")

# 乱数によるデータ作成
np.random.seed(0)
X0 = np.random.randn(100, 2)
X1 = np.random.randn(100, 2) + np.array([2.5, 3])
y = np.array([1] * 100 + [-1] * 100)

X = np.r_[X0 ,X1]

# 作成したモデルのインスタンス
model = svm.SVC()
model.fit(X, y)

xmin, xmax = X[:, 0].min(), X[:, 0].max()
ymin, ymax = X[:, 1].min(), X[:, 1].max()

# 点群の描画
plt.scatter(X0[:, 0], X0[:, 1], color="k", marker="+")
plt.scatter(X1[:, 0], X1[:, 1], color="k", marker="*")


def f(model, x):
    return (-model.w0_ - model.w_[0] * x) / model.w_[1]


xmesh, ymesh = np.meshgrid(
    np.linspace(xmin, xmax, 200), 
    np.linspace(ymin, ymax, 200))

Z = model.predict(np.c_[xmesh.ravel(), ymesh.ravel()]).reshape(xmesh.shape)

# 直線の描画
plt.contour(xmesh, ymesh, Z, levels=[0], colors="k")

print("正しく分類できた数:", (model.predict(X) == y).sum())
plt.show()

