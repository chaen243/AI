from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.utils import to_categorical


#acc- 0.77이상


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts= True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],





#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
#x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] ,x_test.shape[2], 3)




#print(x_train.shape)
#print(x_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)




# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

# print(x_train.shape[0]) #60000

#스케일링1-1 
#x_train = x_train/255. #1.0 0.0
#x_test = x_test/255.

# loss: 0.19173261523246765
# acc: 0.9474999904632568

#스케일링 1-2
# x_train = (x_train - 127.5)/127.5
# x_test = (x_test - 127.5)/127.5

# loss: 0.19397728145122528
# acc: 0.9456999897956848

#numpy에서 연산할땐 부동소수점을 만들어 주는게 연산이 빨라지고 성능도 좋아질수있음.
#이미지데이터에서는 민맥스를 많이 사용.
# 스케일링 2-1
# x_train = x_train.reshape(-1, 28*28) #1.0 0.0
# x_test = x_test.reshape(-1, 28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#loss: 0.17799176275730133
#acc: 0.9465000033378601

# # 스케일링2-2
x_train = x_train.reshape(60000, 28*28,3) #1.0 0.0
x_test = x_test.reshape(10000, 28*28,3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# loss: 0.22253981232643127
# acc: 0.9469000101089478
# 정규화 MinMaxScaler
# 일반화 standardScaler



print(x_train.shape, x_test.shape)



# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

print(np.max(x_train), np.min(x_test))

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

#2.모델
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolutional Block (Conv-Conv-Pool-Dropout)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Classifying
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))



input = Input(shape=x_train.shape[1:])
c1 = Conv2D(32, (3,3), activation= 'relu', padding= 'same')(input)
c2 = Conv2D(32, (3, 3), activation= 'relu', padding=' same')(c1)
mp1 = MaxPooling2D(pool_size= (2, 2))(c2)
dr1 = Dropout(0.25)(mp1)
c3 = Conv2D(64, (3,3), activation='relu', padding='same')(dr1)
c4 = Conv2D(64, (3,3), activation='relu', padding='same')(c3)
mp2 = MaxPooling2D(pool_size=(2, 2))(c4)
dr2 = Dropout(0.25)(mp2)
f1 = Flatten()(dr2)
d1 = Dense(512, activation='relu')(f1)
dr3 = Dropout(0.5)(d1)
output1 = Dense(10, activation='softmax')(dr3)
model = Model(inputs = input, outputs = output1)

model.summary()



'''

filepath = "C:\_data\_save\MCP\_k31"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

#x_train = x_train.reshape ( (x_train.shape[0], 32, 32, 3))
#x_test = x_test.reshape ( (x_test.shape[0], 32, 32, 3))

#3. 컴파일, 훈련
model.compile( loss= 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 299, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)

start_time = time.time()
model.fit(x_train, y_train, batch_size=500, verbose=2, epochs= 500, validation_data=(x_valid,y_valid),shuffle=True, callbacks= [es,mcp])
end_time =time.time()

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , end_time - start_time, "초" )


'''