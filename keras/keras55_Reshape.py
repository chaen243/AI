# 원핫, 모델 완성!
# acc 0.985 이상


import numpy as np
from keras.datasets import mnist
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape, Conv1D, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(-1, 28*28) 
x_test = x_test.reshape(-1, 28*28)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)



x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

#2.모델
model = Sequential()
model.add(Dense(9, input_shape=(28,28,1)))          # (None, 28, 28, 9)         18
model.add(Conv2D(10, (3,3)))                        # (None, 26, 26, 10)        820
model.add(Reshape(target_shape=(26*26, 10)))        # (None, 676, 10)           0 
model.add(Conv1D(15, 4))                            # (None, 673, 15)           615
model.add(LSTM(8, return_sequences=True))           # (None, 673, 8)            768
model.add(Conv1D(14, 2))                            # (None, 672, 14)           238
model.add(Dense(units=8))                           # (None, 672, 8)            120
model.add(Dense(7, input_shape=(8,)))               # (None, 672, 7)            63
model.add(Flatten())                                # (None, 4704)              0
model.add(Dense(6))                                 # (None, 6)                 28230
model.add(Dense(10, activation= 'softmax'))         # (None, 10)                70

model.summary()

#
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 27, 27, 30)        150

#  conv2d_1 (Conv2D)           (None, 26, 26, 20)        2420

#  conv2d_2 (Conv2D)           (None, 25, 25, 10)        810

#  flatten (Flatten)           (None, 6250)              0

#  dense (Dense)               (None, 500)               3125500

#  dropout (Dropout)           (None, 500)               0

#  dense_1 (Dense)             (None, 20)                10020

#  dense_2 (Dense)             (None, 10)                210


# filepath = "C:\_data\_save\MCP\_k55"


# #3. 컴파일, 훈련
# model.compile( loss= 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
# es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50, verbose = 0, restore_best_weights= True)
# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True,filepath= False )

# start_time = time.time()
# model.fit( x_train, y_train, batch_size=216, verbose=2, epochs= 200, validation_split=0.2,callbacks=[es,mcp])
# end_time =time.time()

# #4. 평가, 예측

# results = model.evaluate(x_test,y_test)
# print('loss:', results[0])
# print('acc:',  results[1])
# print('걸린시간 :' , end_time - start_time, "초" )

# # loss: 0.16547049582004547
# # acc: 0.9526000022888184
# # 걸린시간 : 26.87352156639099 초
