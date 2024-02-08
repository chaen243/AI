#keras18 copy
#레거시한 머신러닝~~

import numpy as np
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
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping, ModelCheckpoint



#1. 데이터

X, y = load_iris(return_X_y=True)
print(X.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size= 0.8,  shuffle= True, random_state= 11, stratify= y)



#2. 모델구성

allAlgorithms = all_estimators(type_filter='classifier')
#allAlgorithms = all_estimators(type_filter='regressor')

print('allAlgorithms: ', allAlgorithms) #회귀모델의 갯수 : 55
#print("분류모델의 갯수 :", len(allAlgorithms)) #분류모델의 갯수 : 41
#print("회귀모델의 갯수 :", len(allAlgorithms)) 

for name, algorithm in allAlgorithms:
    try: 
       #2. 모델
        model = algorithm()
        filepath = 'C:/_data/_save/MCP/_m04/'
        
        #3. 훈련
        #mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)
        
        model.fit(x_train, y_train)#, callbacks= [mcp])
        
        acc = model.score(x_test, y_test)
        print(name, '의 정답률은 : ', acc)
    except:
        print(name,  '은 에러!')
        #continue


#3. 컴파일, 훈련





#4. 평가, 예측

#results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
#print("model.score: ", results)

#y_predict = model.predict(x_test)
#print(y_predict)



#acc = accuracy_score(y_predict, y_test)
#print("acc :", acc)


