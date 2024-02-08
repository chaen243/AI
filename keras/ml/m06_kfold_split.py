import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold
import pandas as pd

#1. 데이터

datasets = load_iris()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)


n_splits=3
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수
#kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

for train_index, val_index in kfold.split(df):
    print("=================================")
    print(train_index, "\n", val_index)
    print("훈련데이터 갯수:",len(train_index),
          "검증데이터 갯수:",len(val_index))


'''
#2. 모델구성
model = SVC()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))

#4. 예측

'''
