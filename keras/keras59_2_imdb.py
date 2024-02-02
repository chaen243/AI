#영화리뷰 이진분류

from keras.datasets import imdb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Flatten, Embedding
from keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print(x_train.shape, y_train.shape) #(25000,) (25000,)
print(x_test.shape, y_test.shape) #(25000,) (25000,)
print(len(x_train[0]), len(x_test[0])) #218 68
print(y_train[:20]) #[1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]
print(np.unique(y_train, return_counts=True)) #(array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))

label_num = max(len(np.unique(y_train)),len(np.unique(y_test)))

word_max = max([max(i) for i in x_train] + [max(i) for i in x_test])
print(word_max) #9999
word_size = word_max +1
print(word_size) #10000

print(type(x_train)) #<class 'numpy.ndarray'>

print("영화리뷰의 최대길이 : ", max(len(i) for i in x_train)) #2494
print("영화리뷰의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) #238.71364

########전처리##########

from keras.utils import pad_sequences

x_train = pad_sequences(x_train, padding= 'pre', maxlen=250, truncating= 'pre')
x_test = pad_sequences(x_test, padding= 'pre', maxlen=250, truncating= 'pre')

#print(x_train.shape) #(25000, 250)
#print(x_test.shape) #(25000, 250)


#2. 모델구성

model = Sequential()
model.add(Embedding(word_size, 16, input_length=250))
model.add(Conv1D(256, kernel_size=2, activation= 'relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['acc'] )
hist = model.fit(x_train, y_train, epochs= 500, batch_size= 256, validation_data= (x_test,y_test), verbose=2, callbacks= [es] )

#4. 평가, 예측
result = model.evaluate(x_test,y_test,verbose=0)


print('로스:', result[0])
print('acc:', result[1])


import matplotlib.pyplot as plt

plt.figure(figsize= (9,6))
plt.plot(hist.history['val_acc'], c = 'pink', label = 'val_acc', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')

plt.legend(loc = 'upper right')
plt.title("reuters LOSS")
plt.xlabel('epoch')
plt.grid()
plt.show()



# 로스: 2.024529457092285
# acc: 0.8579999804496765