#https://www.kaggle.com/datasets/playlist/men-women-classification

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time as tm
startTime = tm.time()
xy_traingen =  ImageDataGenerator(
    rescale=1./255,  
)

xy_testgen = ImageDataGenerator(
    rescale=1./255
)

path_train ='C:\\_data\\kaggle\\men-women\\data\\'
path_test ='C:\\_data\\kaggle\\men-women\\test\\'

xy_train = xy_traingen.flow_from_directory(
    path_train,
    batch_size=99999,
    target_size=(150,150),
    class_mode='binary',
    shuffle=True
)

xy_test = xy_testgen.flow_from_directory(
    path_test,
    batch_size=99999,
    target_size=(150,150),
    class_mode='binary'
)
print(xy_test[0][1].shape)
unique, count =  np.unique(xy_test[0][1] , return_counts=True)
print(unique, count)

np_path = 'c:/_data/_save_npy/'
np.save(np_path + 'keras39_5_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras39_5_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras39_5_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'keras39_5_y_test.npy', arr=xy_test[0][1])