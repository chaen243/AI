###########################
## 그래프 그리기
##1. value_counts xx
##2. np.unique xx

##############groupby 사용, count()쓰기

#plt.bar 그리기

#힌트
#데이터갯수(y축)

#############

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#1. 데이터 
path = "C:\\_data\\daicon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

submit_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submit_csv.shape) #(1000, 2)

print(train_csv.columns)
#Index(['quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    #   dtype='object')
    
    
    
#========라벨==========    
x = train_csv.drop(['quality'], axis=1) 
y = train_csv['quality']-3

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
print(x)

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0
    

import matplotlib.pyplot as plt
# plt.boxplot(x)
# plt.show()

###########################
## 그래프 그리기
##1. value_counts xx
##2. np.unique xx

##############groupby 사용, count()쓰기

#plt.bar 그리기

#힌트
#데이터갯수(y축)

#############

y1 = y.groupby(y).sum()
print(y1)

plt.bar(y1, y)
plt.show()