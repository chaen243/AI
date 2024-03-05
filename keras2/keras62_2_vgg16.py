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

#1. 데이터

vgg16 = VGG16(#weights='imagenet', 
              include_top=False,
              input_shape= (32, 32, 3))
vgg16.trainable = False


model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation= 'softmax'))

model.summary()
