import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.callbacks import EarlyStopping

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])
             
y = np.array([4,5,6,7,8,9,10])      

print(x.shape)    #(7, 3)

print(y.shape)    #(7,)

x = x.reshape(7,3,1)

print(x.shape)     #(7, 3, 1)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(units= 10, input_shape=(3,1), activation='relu')) # <- timesteps, features

# 파라미터 갯수 = UNITS * (UNITS + BIAS+ features )

#rnn model inputshape = 3-D tensor with shape (batch_size, timesteps, features).


#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 205 (820.00 Byte)
# Trainable params: 205 (820.00 Byte)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________
model.add(Dense(7,activation= 'relu'))
model.add(Dense(1))

model.summary()


#x가 과하게 크거나 과하게 작으면 기울기 소실 / memory bomb이 있을수 있음. 그래서 tanh 함수를 사용해서 -1~1사이의 수로 만들어줌.
#time step이 길어질수록 최초 time step에 대한 소실이 일어나서 영향을 미치지 못함.


# #3. 컴파일, 훈련
# es = EarlyStopping(monitor= 'loss', mode= 'auto', patience = 100, verbose= 0, restore_best_weights= True )
# model.compile(loss= 'mse', optimizer= 'adam', metrics= 'acc')
# model.fit(x,y,epochs=2500, callbacks= [es], batch_size= 1)

# #4. 평가, 훈련
# results = model.evaluate(x,y)
# print ('loss:', results)
# y_predict = np.array([8,9,10]).reshape(1,3,1)
# y_predict = model.predict(y_predict)
# print('예측값 :', y_predict)

# loss: [1.324871501395819e-08, 0.0]
# 예측값 : [[11.000236]]# 1/1 [==============================] - 0s 89ms/step


