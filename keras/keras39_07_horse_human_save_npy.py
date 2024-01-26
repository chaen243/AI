#다중분류로 풀기
#사진사이즈300*300

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten,MaxPooling2D,Dropout
from sklearn.model_selection import train_test_split
import pandas as pd
import time

#1. 데이터
start_time = time.time()

path = 'C:\\_data\\image\\horse_human\\'

train_datagen = ImageDataGenerator(rescale=1./255,)



xy_train = train_datagen.flow_from_directory(
    path, target_size= (300, 300),
    batch_size= 1027,
    class_mode='categorical',
    shuffle=False
) #Found 1027 images belonging to 2 classes.

#print(xy_train) 
#<keras.preprocessing.image.DirectoryIterator object at 0x000002C9847D3520>

#print(xy_train.next())

#print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
#print(type(xy_train[0])) #<class 'tuple'>
#print(type(xy_train[0][0])) #<class 'numpy.ndarray'>

#x = np.array(np.arange(0,27)).reshape(3,3,3,1)
#y = np.array(np.arange(0,27)).reshape(3,3,3,1)

#x = xy_train[0][0]
#y = xy_train[0][1]

x=[]
y=[]
failed_i = []

for i in range(len(xy_train)):
    try:
        xy_data = xy_train.next()
        new_x = xy_data[0]
        new_y = xy_data[1]
        if i==0:
            x = np.array(new_x)
            y = np.array(new_y)
            continue
        
        x = np.vstack([x,new_x])
        y = np.vstack([y,new_y])
        print("i:", i)
    except:
        print("failed i: ",i)
        failed_i.append(i)        
            

save_path = "C:\\_data\\_save_npy\\"
np.save(save_path + "horse_human_x.npy", arr=x)
np.save(save_path + "horse_human_y.npy", arr=y)


      
