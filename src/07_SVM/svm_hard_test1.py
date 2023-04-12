#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import svm_hard

plt.axes().set_aspect("equal")

# 乱数によるデータ作成
np.random.seed(0)
X0 = np.random.randn(20, 2)
X1 = np.random.randn(20, 2) + np.array([5, 5])
y = np.array([1] * 20 + [-1] * 20)

X = np.r_[X0 ,X1]

# 作成したモデルのインスタンス
model = svm_hard.SVC()
model.fit(X, y)

# 点群の描画
plt.scatter(X0[:, 0], X0[:, 1], color="k", marker="+")
plt.scatter(X1[:, 0], X1[:, 1], color="k", marker="*")


def f(model, x):
    return (-model.w0_ - model.w_[0] * x) / model.w_[1]


x1 = -0.2
x2 = 6

# 直線の描画
plt.plot([x1, x2], [f(model, x1), f(model, x2)], color="k")

# サポートベクタに丸を付ける
plt.scatter(X[model.a_ != 0, 0], X[model.a_ != 0, 1], s=200,
            color=(0, 0, 0, 0), edgecolor="k", marker="o")

plt.show()


# In[ ]:




