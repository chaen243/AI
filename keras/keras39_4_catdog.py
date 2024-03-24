#Train Test 를 분리해서 해보기
#불러오는데 걸리는 시간.

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import os


image_path = 'C:\\_data\\kaggle\\catdog\\Test\\test1\\'
path = 'C:\\_data\\kaggle\\catdog\\'
np_path = 'C:/_data/_save_npy/'
x= np.load(np_path + 'keras39_3_xx_train.npy')
y = np.load(np_path + 'keras39_3_yy_train.npy')
test = np.load(np_path + 'keras39_3_ttest.npy')

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2354, stratify=y)
hist = []

model = Sequential()
model.add(Conv2D(32,(3,3),padding='valid',strides=2,input_shape=x_train.shape[1:]))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),padding='valid',strides=2))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(32,(2,2),padding='valid',activation='relu'))
model.add(Conv2D(32,(2,2),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(2048,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# compile & fit
start_time = time.time()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='auto',patience=10,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=48,validation_data=(x_test,y_test),verbose=2,callbacks=[es])
end_time = time.time()



# evaluate & predict
loss = model.evaluate(x_test,y_test)
y_predict = np.round(model.predict(test)).flatten()

print(y_predict.shape)

print(f"LOSS: {loss[0]:.6f}\nACC:  {loss[1]:.6f}")
model.save(path+f"model_save\\acc_{loss[1]:.6f}.h5")

forder_dir = path+"test\\test1"
id_list = os.listdir(forder_dir)
for i, id in enumerate(id_list):
    id_list[i] = int(id.split('.')[0])

for id in id_list:
    print(id)

y_submit = pd.DataFrame({'id':id_list,'Target':y_predict})
print(y_submit)
y_submit.to_csv(path+f"submit\\acc_{loss[1]:.6f}.csv",index=False)


import matplotlib.pyplot as plt
if hist != []:
    plt.title("Cat&Dog CNN")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.plot(hist.history['val_acc'],label='val_acc',color='red')
    plt.plot(hist.history['acc'],label='acc',color='blue')
    # plt.plot(hist.history['val_loss'],label='val_loss',color='red')
    # plt.plot(hist.history['loss'],label='loss',color='blue')
    plt.legend()
    plt.show()
