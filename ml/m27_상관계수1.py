#keras18 copy
#레거시한 머신러닝~~

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC #softvector machine
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
#from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# class CustomXGBClassifier(XGBClassifier):
#     def __str__(self):
#         return 'XGBClassifier()'
    
# aaa = CustomXGBClassifier()
#print(aaa)   #XGBClassifier()

#1. 데이터

# X, y = load_iris(return_X_y=True)
# print(X.shape, y.shape) #(150, 4) (150,)

datasets = load_iris()
x = datasets.data
y = datasets.target    
    
df = pd.DataFrame(x, columns= datasets.feature_names)
print(df)    
df['Target(Y)'] = y
print(df)    

print('=================상관계수 히트맵==============================')    
print(df.corr())

print()
#x끼리의 상관계수가 너무 높을땐 다른 데이터에 영향을 미칠수 있어(과적합확률) 컬럼처리가 필요함.
#y와 상관관계가 있는 데이터가 중요.

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),
            square= True,
            annot=True,
            cbar=True)
plt.show()
#matplotlib 3.7.2 에서는 나옴. 3.8.0에서는 나오지않아 버전 롤백함.
