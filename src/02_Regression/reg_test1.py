#!/usr/bin/env python
# coding: utf-8

# In[35]:


import linearreg_ # 作成したモデル
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

n = 100
scale = 10

np.random.seed(0)
# 要素が乱数のサイズ100*2の行列を生成 0から10までの値
X = np.random.random((n, 2)) * scale
w0 = 1
w1 = 2
w2 = 3
# 線形和 + 乱数
y = w0 + w1 * X[:, 0] + w2  * X[:, 1] + np.random.randn(n)

# 作成したモデルのインスタンス
model = linearreg_.LinearRegression()
model.fit(X, y)
print("係数：", model.w_)
print("(1, 1)に対する予測値：", model.predict(np.array([1, 1])))

xmesh, ymesh = np.meshgrid(
  np.linspace(0, scale, 20),
  np.linspace(0, scale, 20)
  )

zmesh = (
  model.w_[0]
  + model.w_[1] * xmesh.ravel()
  + model.w_[2] * ymesh.ravel()
 ).reshape(xmesh.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color="k")
ax.plot_wireframe(xmesh, ymesh, zmesh, color="r")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




