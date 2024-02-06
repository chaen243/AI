import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Concatenate
from sklearn.metrics import r2_score

#1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T  #ex) 삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T # ex) 원유, 환율, 금시세
x3_datasets = np.array([range(100), range(301, 401), range(77, 177), range(33, 133)]).T # ex) 원유, 환율, 금시세



print(x1_datasets.shape, x2_datasets.shape) #(100, 2) (100, 3)

y1 = np.array(range(3001, 3101)) #비트코인 종가
y2 = np.array(range(13001, 13101)) #이더리움 종가



x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2, train_size=0.7, random_state=123 )

##################################################################

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

#2-3. 모델
input111 = Input(shape=(4,))
dense111 = Dense(512, activation= 'relu', name='bit111')(input111)
dense112 = Dense(256, activation= 'relu', name='bit112')(dense111)
dense113 = Dense(128, activation= 'relu', name='bit113')(dense112)
output111 = Dense(5, activation= 'relu', name='bit114')(dense113)


#2-4. concatenate
merge1 = concatenate([output1,output11, output111], name= 'mg1') #두개이상은 리스트!/얘도 레이어!
#concatenate로 두개의 모델을 엮어준것!
#행(총데이터)의 갯수는 다 맞춰줘야함
merge2 = Dense(512, name= 'mg2')(merge1)
merge3 = Dense(256, name= 'mg3')(merge2)
merge4 = Dense(128, name= 'mg4')(merge3)
merge5 = Dense(64, name= 'mg5')(merge4)
last_output1 = Dense(1, name= 'last')(merge5)


merge11 = Dense(512, name= 'mg2')(merge1)
merge12 = Dense(256, name= 'mg3')(merge2)
merge13 = Dense(128, name= 'mg4')(merge3)
merge14 = Dense(64, name= 'mg5')(merge4)
last_output2 = Dense(1, name= 'last2')(merge5)







model = Model(inputs= [input1, input11, input111], outputs= [last_output1,last_output2])

model.summary()
'''
#3. 모델, 컴파일

model.compile(loss= 'mse',optimizer= 'adam')
model.fit([x1_train,x2_train, x3_train], [y1_train, y2_train], epochs=500, batch_size=20)

#4. 평가, 예측
result = model.evaluate([x1_test,x2_test,x3_test],[y1_test, y2_test])
print('로스:', result)
y_predict = model.predict([x1_test,x2_test, x3_test])
print('y의 예측값:',y_predict)


r2_1 = r2_score(y1_test,y_predict[0])
r2_2 = r2_score(y2_test,y_predict[1])
print ('r2-1:', r2_1 )
print ('r2-2:', r2_2 )


#로스: [3.688035249710083, 0.2930279076099396, 3.395007371902466]
#r2-1: 0.9996222871896694
#r2-2: 0.9956238377846455

'''