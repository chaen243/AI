#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


#1. 데이터
path = "c:\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)
test_csv = pd.read_csv(path +"test.csv",index_col=0  ).drop(['Pregnancies', 'DiabetesPedigreeFunction'], axis=1)

test = train_csv['SkinThickness']
for i in range(test.size):
      if test[i] == 0:
         test[i] =test.mean()
        
#train_csv['BloodPressure'] = test


print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv") 
print(submission_csv)

print(train_csv.shape)      #(652, 9)
print(test_csv.shape)       #(116, 8)
print(submission_csv.shape) #(116, 2)

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

model= Sequential()
model.add(Dense(1024, input_dim= 6, activation= 'relu'))
model.add(Dense(512 )) 
model.add(Dense(10, )) 
model.add(Dense(1, activation= 'sigmoid'))    
   
   
#3. 컴파일, 훈련
model.compile (loss = 'binary_crossentropy', optimizer= 'adam', metrics= ['acc']) 

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor= 'acc', mode= 'max',
                   patience=500, verbose=0, restore_best_weights= True) #es는 verbose2가 es 정보를 보여줌.
start_time = time.time()
hist= model.fit(x_train, y_train, epochs= 5000, batch_size=1, validation_split= 0.4, verbose=2, callbacks= [es] ) #검증모델은 간접적인 영향을 미침.
end_time = time.time()
  
#4. 평가, 예측
loss = model.evaluate(x_test, y_test) #두개를 비교해서 로스를 빼내줌.
y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)
#print(y_submit)
#print(y_submit.shape)
# y_submit = model.predict(x_test)
#print(submission_csv.shape)
print("로스 :", loss)


submission_csv['Outcome'] =  np.around(y_submit) 



import time as tm

def ACC(y_test, y_predict):
    return accuracy_score(y_test, np.around(y_predict))
acc = ACC(y_test, y_predict)
print("정확도 : ", acc)


ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}{acc:.3f}.csv", index=False)

print("걸린 시간 :", round(end_time - start_time, 2), "초")

####### submission.csv 만들기 (count컬럼에 값만 넣어주면 됨) #####






#로스 : 3175.002197265625
#R2 스코어 : 0.5593716340440571
#RMSE :  56.347159447801296

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']= False
plt.figure(figsize= (9,6))
#plt.plot(hist.history['loss'], c = 'red', label = 'loss', marker = '.')
plt.plot(hist.history['val_acc'], c = 'blue', label = 'val_loss', marker = '.')
plt.plot(hist.history['acc'], color= 'violet', label = 'acc', marker= '.')
plt.legend(loc = 'upper right')
plt.title("당뇨병 LOSS")
plt.xlabel('epoch')
plt.grid()
plt.show()

#로스 : [1.0998154878616333, 0.6938775777816772]
#정확도 :  0.6938775510204082