import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Concatenate
from sklearn.metrics import r2_score

#1. 데이터
x = np.array([range(100), range(301,401)]).T  #ex) 삼성 종가, 하이닉스 종가




y1 = np.array(range(3001, 3101)) #비트코인 종가
y2 = np.array(range(13001, 13101)) #이더리움 종가




x_train, x_test,  y1_train, y1_test, y2_train, y2_test = train_test_split(
    x, y1, y2, train_size=0.7, random_state=123 )

##################################################################

# print(x1_train.shape, x2_train.shape, y_train.shape) #(70, 2) (70, 3) (70,)
# #train_test_split은 데이터가 몇개든 다 잘라줌

#2-1. 모델

input1 = Input(shape=(2, ))
dense1 = Dense(512, activation= 'relu', name='bit1')(input1)
dense2 = Dense(256, activation= 'relu', name='bit2')(dense1)
dense3 = Dense(128, activation= 'relu', name='bit3')(dense2)
output1 = Dense(1, activation= 'relu', name='bit4')(dense3)





#2-4. concatenate
#concatenate로 두개의 모델을 엮어준것!
#행(총데이터)의 갯수는 다 맞춰줘야함
output1 = Dense(1024, activation= 'relu', name='bit4')(dense3)
merge2 = Dense(512, name= 'mg2')(output1)
merge3 = Dense(256, name= 'mg3')(merge2)
merge4 = Dense(128, name= 'mg4')(merge3)
merge5 = Dense(64, name= 'mg5')(merge4)
last_output1 = Dense(1, name= 'last')(merge5)

merge11 = Dense(512, name= 'mg2')(output1)
merge12 = Dense(256, name= 'mg3')(merge2)
merge13 = Dense(128, name= 'mg4')(merge3)
merge14 = Dense(64, name= 'mg5')(merge4)
last_output2 = Dense(1, name= 'last2')(merge5)

model = Model(inputs= input1, outputs= [last_output1,last_output2])


#model.summary()

#3. 모델, 컴파일

model.compile(loss= 'mse',optimizer= 'adam')
model.fit(x_train, [y1_train,y2_train], epochs=500, batch_size=20)

#4. 평가, 예측
result = model.evaluate(x_test,[y1_test,y2_test])
print('로스:', result)
y_predict = model.predict(x_test)
print('y의 예측값:',y_predict)
r2_1 = r2_score(y1_test,y_predict[0])
r2_2 = r2_score(y2_test,y_predict[1])
print ('r2-1:', r2_1 )
print ('r2-2:', r2_2 )

#로스: [17.1422061920166, 0.05674951151013374, 17.08545684814453]
# r2-1: 0.9999268499125247
# r2-2: 0.9779768547770894

