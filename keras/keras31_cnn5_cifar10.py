from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.utils import to_categorical


#acc- 0.77이상


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts= True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],





#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
#x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] ,x_test.shape[2], 3)




#print(x_train.shape)
#print(x_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]


#스케일링1-1 

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.



#스케일링 1-2
# x_train = (x_train - 127.5)/127.5
# x_test = (x_test - 127.5)/127.5



#numpy에서 연산할땐 부동소수점을 만들어 주는게 연산이 빨라지고 성능도 좋아질수있음.
#이미지데이터에서는 민맥스를 많이 사용.
# 스케일링 2-1
# x_train = x_train.reshape(-1, 28*28) #1.0 0.0
# x_test = x_test.reshape(-1, 28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)




# # 스케일링2-2
# x_train = x_train.reshape(-1, 28*28) #1.0 0.0
# x_test = x_test.reshape(-1, 28*28)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# 정규화 MinMaxScaler
# 일반화 standardScaler


print(x_train.shape, x_test.shape)



# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

print(np.max(x_train), np.min(x_test))

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

#2.모델
model = Sequential()
# model.add(Conv2D(32, (3,3),
#                  input_shape= (32, 32, 3), activation= 'relu')) #첫 아웃풋 = filter
# # shape = (batch_size, rows, columns, channels)
# # shape = (batch_size, heights, widths, channels)
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3,3), activation= 'relu')) 
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (2,2), activation= 'relu')) 
# model.add(Conv2D(32, (2,2), activation= 'relu'))
# model.add(MaxPooling2D(pool_size=(2, 2))) 
# model.add(Flatten()) #(n,22*22*15)의 모양을 펴져있는 모양으로 변형.
# # shape = (batch_size(=model.fit의 batch_size와 같음.), input_dim) 
# model.add(Dense(10, activation= 'softmax'))
# Convolutional Block (Conv-Conv-Pool-Dropout)
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolutional Block (Conv-Conv-Pool-Dropout)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Classifying
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))





filepath = "C:\_data\_save\MCP\_k31"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

#x_train = x_train.reshape ( (x_train.shape[0], 32, 32, 3))
#x_test = x_test.reshape ( (x_test.shape[0], 32, 32, 3))

#3. 컴파일, 훈련
model.compile( loss= 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 299, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)

start_time = time.time()
model.fit(x_train, y_train, batch_size=500, verbose=2, epochs= 500, validation_data=(x_valid,y_valid),shuffle=True, callbacks= [es,mcp])
end_time =time.time()

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , end_time - start_time, "초" )

# loss: 0.6674924492835999
# acc: 0.7788000106811523
# 걸린시간 : 459.2906882762909 초
