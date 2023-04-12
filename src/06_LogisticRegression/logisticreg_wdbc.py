#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 検査数値と乳がんの診断結果についてのデータにロジスティック回帰を適用したプログラム
import logisticreg # 作成したモデル
import numpy as np
import csv

n_test = 100
X = []
y = []

# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
with open("wdbc.data") as fp:
    for row in csv.reader(fp):
        if row[1] == "B":
            y.append(0)
        else:
            y.append(1)
        X.append(row[2:])

y = np.array(y, dtype=np.float64)
X = np.array(X, dtype=np.float64)

train_y = y[:-n_test]
train_X = X[:-n_test]
test_y = y[-n_test:]
test_X = X[-n_test:]

model = logisticreg.LogisticRegression(tol=0.01)
model.fit(train_X, train_y)

y_predict = model.predict(test_X)
n_hits = (test_y == y_predict).sum() # 予想があたってい｀る物の数
print("Accuracy: {}/{} = {}".format(n_hits, n_test, n_hits / n_test))


# In[ ]:




