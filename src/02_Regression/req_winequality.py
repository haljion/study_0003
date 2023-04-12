#!/usr/bin/env python
# coding: utf-8

# In[6]:


# 線形回帰の実践的な例
import linearreg_ # 作成したモデル
import numpy as np
import csv

# データ読み込み
Xy = []

# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
with open("winequality-red.csv") as fp:
    for row in csv.reader(fp, delimiter=";"):
        Xy.append(row)

# 文字列のリストのリスト→数値の2次元配列
Xy = np.array(Xy[1:], dtype=np.float64)

# 訓練データとテストデータに分割する
np.random.seed(0)
np.random.shuffle(Xy)
train_X = Xy[:-1000, :-1]
train_y = Xy[:-1000, -1]
test_X = Xy[-1000:, :-1]
test_y = Xy[-1000:, -1]

# 学習
model = linearreg_.LinearRegression()
model.fit(train_X, train_y)

# テスト用データにモデルを適用
y = model.predict(test_X)

print("最初の5つの正解と予測値：")
for i in range(5):
    print("{:1.0f} {:5.3f}".format(test_y[i], y[i]))
    
print()
# RMSE(Root of Mean Square Error): 平均二乗誤差のルート
print("RMSE:", np.sqrt(((test_y - y)**2).mean()))


# In[ ]:





# In[ ]:




