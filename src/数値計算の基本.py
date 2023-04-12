#!/usr/bin/env python
# coding: utf-8

# In[6]:


# 約17桁程度までは正確に計算されるが、それ以上になると誤差が出る
# 誤差による無限ループなどに注意
s = 0
for i in range(1000):
    s+= 0.001
s


# In[15]:


# 条件式などに使用する場合は、十分に小さい数ε(イプシロン)と誤差を比較する手法が主流
eps = 1e-10
s = 0
while abs(s - 1.) > eps:
    print(s)
    s += 0.1


# In[22]:


import numpy as np

# 2次方程式の解を求める関数
def qeq(a, b, c):
    # sqrt: 平方根
    d = np.sqrt(b**2 - 4 * a * c)
    return ((-b + d) / (2 * a), (-b - d) / (2 * a))

# x^2 + 5x + 6
qeq(1, 5, 6)


# In[9]:





# In[ ]:




