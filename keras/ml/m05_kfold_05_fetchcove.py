import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression#분류!
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time
import tensorflow as tf
#1. 데이터

start_time = time.time()

datasets = fetch_covtype()

x = datasets.data
y = datasets.target
#print(x.shape, y.shape) #(581012, 54) (581012,)
#print(pd.value_counts(y))
#print(np.unique(y, return_counts= True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],)
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score


n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수

#2. 모델구성
model = KNeighborsClassifier()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))

#model.score : 0.7131677987883238

#Linear
# model.score : 0.5243826877180099
# acc : 0.5243826877180099
# 걸린시간 : 348.6212000846863 초

# KNeighborsClassifier
# acc : [0.96832268 0.96858085 0.96873548 0.96958744 0.96873548] 
#  평균 acc : 0.9688

#Stratified
# acc : [0.96861527 0.96858945 0.96864942 0.96855476 0.96852894] 
#  평균 acc : 0.9686