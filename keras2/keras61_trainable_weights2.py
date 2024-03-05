import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__) #2.9.0

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
# model.add(Dense(2))
# model.add(Dense(2))
model.add(Dense(1))


#############################################################
model.trainable = False #★★★ 훈련을 시키지 않는다!
#model.trainable = True #디폴트! 
#############################################################

print(model.weights)
print('='*100)

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam')
model.fit(x, y, batch_size = 1, epochs=1000, verbose=0)

#4. 평가, 예측
y_pred = model.predict(x)
print(y_pred)