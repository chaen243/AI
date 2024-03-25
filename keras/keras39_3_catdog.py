#Train Test 를 분리해서 해보기
#불러오는데 걸리는 시간.

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

path_train ='c:/_data/kaggle/catdog/train/'
path_test ='C://_data//kaggle//catdog//Test//'

BATCH = int(2000)

xy_train = xy_traingen.flow_from_directory(
    path_train,
    batch_size=BATCH,
    target_size=(100,100),
    class_mode='binary',
    shuffle=True
)

xy_test = xy_testgen.flow_from_directory(
    path_test,
    batch_size=BATCH,
    target_size=(100,100),
    class_mode='binary',
    shuffle= False
)



x = []
y = []
failed_i = []

for i in range(int(20000 / BATCH)):
    try: 
        xy_data = xy_train.next()
        new_x = xy_data[0]
        new_y = xy_data[1]
        if i==0:
            x = np.array(new_x)
            y = np.array(new_y)
            continue
        
        x = np.vstack([x,new_x])
        y = np.hstack([y,new_y])
        print("i: ",i)
        print(f"{x.shape=}\n{y.shape=}")
    except:
        print("failed i: ",i)
        failed_i.append(i)
        
print(failed_i) # [70, 115]

print(f"{x.shape=}\n{y.shape=}")    # x.shape=(1000, 200, 200, 3) y.shape=(1000,)


np_path = 'c:/_data/_save_npy/'
np.save(np_path + 'keras39_3_xx_train.npy', arr=x)
np.save(np_path + 'keras39_3_yy_train.npy', arr=y)
np.save(np_path + 'keras39_3_ttest.npy', arr=xy_test[0][0])

