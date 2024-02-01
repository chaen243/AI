import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Concatenate
#concatenate = 함수
#Concatenate = 클래스!



#1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T  #ex) 삼성 종가, 하이닉스 종가
X2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T # ex) 원유, 환율, 금시세

print(x1_datasets.shape, X2_datasets.shape) #(100, 2) (100, 3)

y = np.array(range(3001, 3101)) #비트코인 종가

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, X2_datasets, y, train_size=0.7, random_state=123 )

# print(x1_train.shape, x2_train.shape, y_train.shape) #(70, 2) (70, 3) (70,)
# #train_test_split은 데이터가 몇개든 다 잘라줌

#2-1. 모델

input1 = Input(shape=(2, ))
dense1 = Dense(512, activation= 'relu', name='bit1')(input1)
dense2 = Dense(256, activation= 'relu', name='bit2')(dense1)
dense3 = Dense(128, activation= 'relu', name='bit3')(dense2)
output1 = Dense(10, activation= 'relu', name='bit4')(dense3)

#model = Model(inputs= input1, outputs=output1)
#model.summary()

#2-2. 모델
input11 = Input(shape=(3, ))
dense11 = Dense(512, activation= 'relu', name='bit11')(input11)
dense12 = Dense(256, activation= 'relu', name='bit12')(dense11)
dense13 = Dense(128, activation= 'relu', name='bit13')(dense12)
output11 = Dense(5, activation= 'relu', name='bit14')(dense13)

#model2 = Model(inputs= input11, outputs=output11)

#model2.summary()

#name은 성능에 영향을 주지 않는다. 라벨링 해준것.

#2-3. concatenate

#concatenate로 두개의 모델을 엮어준것!
#행(총데이터)의 갯수는 다 맞춰줘야함


merge1 = Concatenate(name= 'mg1')([output1,output11]) #사용법이 다름!
#merge1 = concatenate([output1,output11], name= 'mg1') #두개이상은 리스트!/얘도 레이어!
merge2 = Dense(512, name= 'mg2')(merge1)
merge3 = Dense(256, name= 'mg3')(merge2)
merge4 = Dense(128, name= 'mg4')(merge3)
merge5 = Dense(64, name= 'mg5')(merge4)
last_output = Dense(1, name= 'last')(merge5)

model = Model(inputs= [input1, input11], outputs= last_output)

#model.summary()

#3. 모델, 컴파일

model.compile(loss= 'mse',optimizer= 'adam')
model.fit([x1_train,x2_train], y_train, epochs=123, batch_size=20)

#4. 평가, 예측
result = model.evaluate([x1_test,x2_test],y_test)
print('로스:', result)
y_predict = model.predict([x1_test,x2_test])
print('y의 예측값:',y_predict)

