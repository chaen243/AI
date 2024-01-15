from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([1,2,3]) #= ([[1],[2],[3]])
y = np.array([1,2,3])

model = Sequential()
model.add(Dense(5, input_shape=(1,)))
model.add(Dense(4, ))
model.add(Dense(2, ))
model.add(Dense(1, ))

model.summary()

#bias 때문에 노드가 하나 더 잡혀서 전산량이 늘어남.
#더하기 연산이기 때문에 큰 영향을 미치지는 않지만 좌우로 움직이는 연산에 영향을 미치기때문에 꼭 필요한 연산임.


# Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 5)                 10

#  dense_1 (Dense)             (None, 4)                 24

#  dense_2 (Dense)             (None, 2)                 10

#  dense_3 (Dense)             (None, 1)                 3