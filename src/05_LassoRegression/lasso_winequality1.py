#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ワインの品質データにラッソ回帰を適用したプログラム
import lasso # 作成したモデル
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

# ハイパーパラメータを変えながら学習させて表示
for lambda_ in [1., 0.1, 0.01]:
    model = lasso.Lasso(lambda_)
    model.fit(train_X, train_y)
    # テスト用データにモデルを適用
    y = model.predict(test_X)
    print("--- lambda = {} ---".format(lambda_))
    print("coefficients:")
    # 出力が疎になっている
    print(model.w_)
    mse = ((y - test_y)**2).mean()
    # MSE: 平均二乗誤差
    print("MSE: {:.3f}".format(mse))


# In[ ]:




