from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import os
import random as rn
import keras
import numpy as np
SEED = 115

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.random.set_seed(SEED)  #텐서 2.9 먹힘 2.12 안먹힘
np.random.seed(SEED)
rn.seed(SEED)

# rn.seed(333)
# tf.random.set_seed(123)
# np.random.seed(321)


print(tf.__version__)  #2.9.0
print(keras.__version__) #2.9.0
print(np.__version__) #1.26.3

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델

model = Sequential()
model.add(Dense(5, input_dim=1))
                #kernel_initializer='zeros', input_dim=1))
model.add(Dense(5))
model.add(Dense(1))
#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x, y, epochs=100, verbose=0)

#4. 평가, 예측
loss = model.evaluate(x, y, verbose=0)
print ('loss', loss)
results= model.predict([4],verbose=0)
print("4:", results)