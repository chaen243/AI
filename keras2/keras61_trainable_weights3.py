#61_2의 결과를 손으로 계산
#0에포


import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(779)
print(tf.__version__) #2.9.01

#1. 데이터
x = np.array([1,2,])
y = np.array([1,2,])

#2. 모델
model = Sequential()
model.add(Dense(2
                , input_dim = 1))
#model.add(Dense(2))
# model.add(Dense(2))
# model.add(Dense(2))
model.add(Dense(1))


#############################################################
model.trainable = False #★★★ 훈련을 시키지 않는다!
#model.trainable = True #디폴트!!
#############################################################

print(model.weights)
print('='*100)

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam')
model.fit(x, y, batch_size = 1, epochs=1000, verbose=0)

#4. 평가, 예측
y_pred = model.predict(x)
print(y_pred)

# [<tf.Variable 'dense/kernel:0' shape=(1, 2) dtype=float32, numpy=array([[ 0.5460421, -0.9101932]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[-0.22245848],
#        [ 0.6616527 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
# ====================================================================================================

# y_pred
# [[-0.7237035]
#  [-1.447407 ]]