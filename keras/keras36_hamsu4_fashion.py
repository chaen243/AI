from keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

#print(np.unique(y_train, return_counts= True))
#([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],

#plt.imshow(x_train[5], 'gray')
#plt.show()



x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#print(x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer


mms = MinMaxScaler(feature_range=(-2,2))
#mms = StandardScaler()
#mms = MaxAbsScaler()
#mms = RobustScaler()



x_train = mms.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
x_test = mms.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
x_valid = mms.transform(x_valid.reshape(-1, x_valid.shape[-1])).reshape(x_valid.shape)

#2. 모델구성

model = Sequential()
model.add(Conv2D(10, (2,2),
                 input_shape= (28, 28, 1), activation= 'swish')) #첫 아웃풋 = filter
# shape = (batch_size, rows, columns, channels)
# shape = (batch_size, heights, widths, channels)
# 통상적으로 conv2D 레이어는 2단이상으로 쌓음. (1단으로는 성능이 잘 안나와..)
model.add(Conv2D(filters=20, kernel_size=(3,3), activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(10, (2,2), activation= 'relu')) 
model.add(Conv2D(20, (2,2), activation= 'relu'))  
model.add(Conv2D(10, (2,2), activation= 'relu')) 
model.add(MaxPooling2D(2,2))
model.add(Flatten()) #(n,22*22*15)의 모양을 펴져있는 모양으로 변형. (행렬연산임으로 연산 가능!)
model.add(Dense(20, activation= 'relu'))
# shape = (batch_size(=model.fit의 batch_size와 같음.), input_dim) 
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation= 'softmax'))


########함수형########


input = Input(shape= x_train.shape[1:])
c1= Conv2D(10, (2,2), activation = 'swish')(input)
c2 = Conv2D(20, (3,3), activation= 'relu')(c1)
mp1 = MaxPooling2D(2,2)(c2)
c3 = Conv2D(10, (2,2), activation= 'relu')(mp1)
c4 = Conv2D(20, (2,2), activation= 'relu')(c3)
c5 = Conv2D(10, (2,2), activation= 'relu')(c4)
mp2 = MaxPooling2D(2,2)(c5)
f1 = Flatten()(mp2)
d1 = Dense(20, activation= 'relu')(f1)
d2 = Dense(10, activation= 'relu')(d1)
d3 = Dense(20, activation= 'relu')(d2)
output = Dense(10, activation= 'softmax')(d3)
model = Model(inputs= input, outputs = output)


model.summary()
'''

filepath = "C:\_data\_save\MCP\_k31"

#3. 컴파일, 훈련
model.compile ( loss = 'categorical_crossentropy', optimizer = 'adam', metrics= 'acc')
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 200, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)

start_time = time.time()
model.fit(x_train, y_train, batch_size = 216, verbose=2, epochs= 600, validation_data= (x_valid, y_valid), callbacks= [es, mcp])
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , end_time - start_time, "초" )
'''