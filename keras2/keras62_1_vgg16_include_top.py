import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__) #2.9.0

from keras.applications import VGG16

#model = VGG16()
# 디폴트 연산량/ (224,224,3)으로 고정되어있음
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0


model = VGG16(#weights='imagenet', 
               include_top=False, #True일때는 (224,224,3)으로 고정되어있음, # False일땐 fully_connected layer 죽임
            #   연산량은 flatten 이후에 더 많음!
              input_shape=(32, 32, 3)
              )


model.summary()

# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0


############include_Top = False ############
#1. FC layer 죽임
#2. input_shape 원하는대로 변경 가능