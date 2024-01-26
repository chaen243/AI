#150*150
import time
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

#1. 데이터

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

path = "C:\\_data\\image\\rps\\"

xy_train = train_datagen.flow_from_directory(
    path,
    target_size=(150, 150),
    batch_size= 9999,
    class_mode='categorical',
    shuffle='True')


x = np.array(np.arange(0,27)).reshape(3,3,3,1)
y = np.array(np.arange(0,27)).reshape(3,3,3,1)

xy = np.vstack([x,y])

x = xy_train[0][0]
y = xy_train[0][1]


save_path = "C:\\_data\\_save_npy\\"
np.save(save_path + "rps_x.npy", arr=x)
np.save(save_path + "rps_y.npy", arr=y)

print('finish')


