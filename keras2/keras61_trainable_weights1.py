import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
#tf.random.set_seed(777)
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

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 6
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3
=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0
_________________________________________________________________


전이학습
'''
print(model.weights)
'''
                     kernel = 가중치!                                    모델 첫 레이어의 가중치 (통상 랜덤값)                                       #통상 첫 값은 0
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.47288632, -0.78825045,  1.2209238 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.1723156 ,  0.5125139 ],
       [ 0.41434443, -0.8537577 ],
       [ 0.5188304 , -0.91461056]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[ 1.1585606],
       [-0.4251585]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''       
print('='*100)
print(model.trainable_weights)   
print('='*100)


print(len(model.weights))        
print('='*100)
print(len(model.trainable_weights))
print('='*100)

#############################################################
#전이학습 (가중치를 그대로 사용함. 좋은 가중치를 만들어 놓았을때 그대로 쓰기 위해!)
#전이학습시에는 라벨갯수, 프레임 사이즈 등에 따라 인풋 아웃풋은 바꿔줘야함.
model.trainable = False #★★★ 훈련을 시키지 않는다!
#############################################################

print('='*100)
print(len(model.weights))
print('='*100)
print(len(model.trainable_weights))
print('='*100)


print('='*100)
print(model.weights)
print('='*100)
print(model.trainable_weights)   
print('='*100)

model.summary()

'''

====================================================================================================
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 6
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3
=================================================================
Total params: 17
Trainable params: 0
Non-trainable params: 17
_________________________________________________________________
'''
