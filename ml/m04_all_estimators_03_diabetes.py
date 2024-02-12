#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LogisticRegressionCV#분류!
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#1. 데이터
path = "c:\\_data\\daicon\\diabetes\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)
test_csv = pd.read_csv(path +"test.csv",index_col=0  ).drop(['Pregnancies', 'DiabetesPedigreeFunction'], axis=1)

test = train_csv['SkinThickness']
for i in range(test.size):
      if test[i] == 0:
         test[i] =test.mean()
        
#train_csv['BloodPressure'] = test


submission_csv = pd.read_csv(path + "sample_submission.csv") 


print(train_csv.columns) #'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
      # 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      

  

x = train_csv.drop(['Outcome','Pregnancies','DiabetesPedigreeFunction'], axis=1)

#print(x)
y = train_csv['Outcome']
#print(y)
from sklearn.model_selection import train_test_split,KFold,cross_val_score

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


#2. 모델구성
model = LogisticRegressionCV()

#3. 컴파일, 훈련

scores = cross_val_score(model, x, y, cv=kfold)
print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))


# acc : [0.70229008 0.73282443 0.77692308 0.77692308 0.80769231] 
#  평균 acc : 0.7593




#LinearSVC
# model.score: 0.6224489795918368
# 정확도 :  0.6224489795918368