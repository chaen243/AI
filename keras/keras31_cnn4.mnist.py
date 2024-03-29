# 원핫, 모델 완성!
# acc 0.985 이상


import numpy as np
from keras.datasets import mnist
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)



#print(x_train)
print(x_train[0])
print(y_train[0]) # 5
print(np.unique(y_train, return_counts= True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
print(pd.value_counts(y_test))


# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

# print(x_train.shape[0]) #60000

#스케일링1-1 
#x_train = x_train/255. #1.0 0.0
#x_test = x_test/255.

# loss: 0.19173261523246765
# acc: 0.9474999904632568

#스케일링 1-2
# x_train = (x_train - 127.5)/127.5
# x_test = (x_test - 127.5)/127.5

# loss: 0.19397728145122528
# acc: 0.9456999897956848

#numpy에서 연산할땐 부동소수점을 만들어 주는게 연산이 빨라지고 성능도 좋아질수있음.
#이미지데이터에서는 민맥스를 많이 사용.
# 스케일링 2-1
# x_train = x_train.reshape(-1, 28*28) #1.0 0.0
# x_test = x_test.reshape(-1, 28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#loss: 0.17799176275730133
#acc: 0.9465000033378601

# # 스케일링2-2
x_train = x_train.reshape(-1, 28*28) #1.0 0.0
x_test = x_test.reshape(-1, 28*28)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# loss: 0.22253981232643127
# acc: 0.9469000101089478
# 정규화 MinMaxScaler
# 일반화 standardScaler


print(x_train.shape, x_test.shape)



y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

print(np.max(x_train), np.min(x_test))

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

#2.모델
model = Sequential()
model.add(Conv2D(30, (2,2),
                 input_shape= (28, 28, 1))) #첫 아웃풋 = filter
# shape = (batch_size, rows, columns, channels)
# shape = (batch_size, heights, widths, channels)
model.add(Conv2D(filters=20, kernel_size=(2,2)))
model.add(Conv2D(10, (2,2))) 
model.add(Flatten()) #(n,22*22*15)의 모양을 펴져있는 모양으로 변형.
model.add(Dense(500))
model.add(Dropout(0.05))
# shape = (batch_size(=model.fit의 batch_size와 같음.), input_dim) 
model.add(Dense(20, activation='swish'))
model.add(Dense(10, activation= 'softmax'))


#batch_size는 전체 행에서 원하는만큼 행을 나눠 훈련) 레이어별로 다르게 지정 불가. (SHAPE ERROR 뜸.)
# model.summary()
# (아웃풋수 * 커널사이즈 * 커널갯수 )+ bias = Param
# (channels * kernel size + bias) * 9 
#_________________________________________________________________     
#  Layer (type)                Output Shape              Param #        
# =================================================================     
#  conv2d (Conv2D)             (None, 27, 27, 9)         45
#(kernel size(4) * channels(1) + bias(1)) * (filter)(9) = 45
#  conv2d_1 (Conv2D)           (None, 25, 25, 10)        820
#(9 * 9 + 1) * 10 = 820
#  conv2d_2 (Conv2D)           (None, 22, 22, 15)        2415
#(16 * 10 + 1) * 15 = 2415
#  flatten (Flatten)           (None, 7260)              0

#  dense (Dense)               (None, 8)                 58088

#  dense_1 (Dense)             (None, 7)                 63

#  dense_2 (Dense)             (None, 6)                 48

#  dense_3 (Dense)             (None, 5)                 35

#  dense_4 (Dense)             (None, 10)                60

# =================================================================
# Total params: 61,574
# Trainable params: 61,574
# Non-trainable params: 0
# _________________________________________________________________



filepath = "C:\_data\_save\MCP\_k31"


#3. 컴파일, 훈련
model.compile( loss= 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)

start_time = time.time()
model.fit( x_train, y_train, batch_size=216, verbose=2, epochs= 200, validation_split=0.2,callbacks=[es,mcp])
end_time =time.time()

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , end_time - start_time, "초" )

# loss: 0.16547049582004547
# acc: 0.9526000022888184
# 걸린시간 : 26.87352156639099 초
