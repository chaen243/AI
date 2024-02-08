#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression#분류!
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

print(np.unique(y, return_counts= True)) #(array([0, 1]), array([424, 228])
print(pd.Series(y).value_counts()) #다 똑가틈
print(pd.DataFrame(y).value_counts()) #다 똑가틈 (dataframe을 쓰면 메모리를 더 씀,행렬데이터는 dataframe만 가능)
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.85, shuffle= False, random_state= 293)  

#2. 모델구성
from sklearn.utils import all_estimators

allAlgorithms = all_estimators(type_filter='classifier')
#allAlgorithms = all_estimators(type_filter='regressor')

print('allAlgorithms: ', allAlgorithms) #회귀모델의 갯수 : 55
#print("분류모델의 갯수 :", len(allAlgorithms)) #분류모델의 갯수 : 41
#print("회귀모델의 갯수 :", len(allAlgorithms)) 

for name, algorithm in allAlgorithms:
    try: 
       #2. 모델
        model = algorithm()
        
        #3. 훈련
        
        model.fit(x_train, y_train)#, callbacks= [mcp])
        
        acc = model.score(x_test, y_test)
        print(name, '의 정답률은 : ', acc)
    except:
        print(name,  '은 에러!')
        #continue


#3. 컴파일, 훈련

model.fit(x_train, y_train)
  
#4. 평가, 예측
loss = model.score(x_test, y_test) #두개를 비교해서 로스를 빼내줌.
y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)
#print(y_submit)
#print(y_submit.shape)
# y_submit = model.predict(x_test)
#print(submission_csv.shape)
print("model.score:" , loss)


submission_csv['Outcome'] =  np.around(y_submit) #(2진분류는 소수점값으로 나옴. 답지에는 0,1로 입력해야하기때문에 반올림 해줘야함)



import time as tm

def ACC(y_test, y_predict):
    return accuracy_score(y_test, np.around(y_predict))
acc = ACC(y_test, y_predict)
print("정확도 : ", acc)


ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}{acc:.3f}.csv", index=False)


####### submission.csv 만들기 (count컬럼에 값만 넣어주면 됨) #####






#로스 : 3175.002197265625
#R2 스코어 : 0.5593716340440571
#RMSE :  56.347159447801296



#로스 : [1.0998154878616333, 0.6938775777816772]
#정확도 :  0.6938775510204082


#LinearSVC
# model.score: 0.6224489795918368
# 정확도 :  0.6224489795918368