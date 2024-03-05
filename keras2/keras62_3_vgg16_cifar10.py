#1 cifar10 모델 완성
#2. 시간 체크

import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten

from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__) #2.9.0

from keras.applications import VGG16
from keras.datasets import cifar10
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import time

#1. 데이터



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)



# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


# x_train = x_train.reshape(50000, 32*32,3) #1.0 0.0
# x_test = x_test.reshape(10000, 32*32,3)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
vgg16 = VGG16(#weights='imagenet', 
              include_top=False,
              input_shape= (32, 32, 3))
vgg16.trainable = False


model = Sequential()
model.add(vgg16)
model.add(Flatten())
#model.add(Dense())
model.add(Dense(120))
model.add(Dense(10))#, activation= '))

model.summary()

#3. 컴파일, 훈련

model.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
es = EarlyStopping(monitor= 'val_loss', mode= 'auto', patience= 1, verbose=2, restore_best_weights= True)

start_time = time.time()
model.fit(x_train, y_train, batch_size=100000, verbose=2, epochs= 1000, validation_split=0.25, shuffle=True, callbacks= [es])
end_time =time.time()

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , end_time - start_time, "초" )


#이전
# loss: 0.6674924492835999
# acc: 0.7788000106811523
# 걸린시간 : 459.2906882762909 초

#동결



# loss: 1.2013143301010132
# acc: 0.5950999855995178
# 걸린시간 : 90.15599870681763 초