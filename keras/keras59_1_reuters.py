from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Flatten, Embedding
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping


(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1001,
                                                         test_split=0.2)
print(x_train)
print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)
print(type(x_train)) #<class 'numpy.ndarray'>
print(y_train) #[ 3  4  3 ... 25  3 25]
print(len(np.unique(y_train))) #46
print(len(np.unique(y_test))) #46

print(np.unique(y_train, return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], dtype=int64), array([  55,  432,   74, 3159, 1949,   17,   48,   16,  139,  101,  124,
#         390,   49,  172,   26,   20,  444,   39,   66,  549,  269,  100,
#          15,   41,   62,   92,   24,   15,   48,   19,   45,   39,   32,
#          11,   50,   10,   49,   19,   19,   24,   36,   30,   13,   21,
#          12,   18], dtype=int64))

print(np.unique(y_test, return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], dtype=int64), array([ 12, 105,  20, 813, 474,   5,  14,   3,  38,  25,  30,  83,  13,
#         37,   2,   9,  99,  12,  20, 133,  70,  27,   7,  12,  19,  31,
#          8,   4,  10,   4,  12,  13,  10,   5,   7,   6,  11,   2,   3,
#          5,  10,   8,   3,   6,   5,   1], dtype=int64))
labels_num = max(len(np.unique(y_train)),len(np.unique(y_test)))
print(labels_num)

#len_list = [len(i) for i in x_train] + [len(i) for i in x_test]
word_max = max([max(i) for i in x_train] + [max(i) for i in x_test])
print(word_max) #1000
word_size = word_max +1
print(word_size)



# print(type(x_train)) #<class 'numpy.ndarray'>
# print(type(x_train[0])) #<class 'list'>
# print(len(x_train[0])) #87
# print(len(x_train[1])) #56
# print("뉴스의 최대길이 : ", max(len(i) for i in x_train)) #2376
#print("뉴스의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) #145.5398574927633

#전처리
from keras.utils import pad_sequences

x_train = pad_sequences(x_train, padding= 'pre', maxlen=150, truncating= 'pre')
x_test = pad_sequences(x_test, padding= 'pre', maxlen=150, truncating= 'pre')

#y원핫 안하면 sparsedategorical_crossentropt

#print(x_train.shape) #(8982, 150)+
#print(x_test.shape)  #(2246, 150)

#word_to_index = reuters.get_word_index()
#print(word_to_index)





#2. 모델구성

model = Sequential()
model.add(Embedding(word_size, 512, input_length=150))
model.add(Conv1D(512, kernel_size=2, input_shape= (150, 1), activation= 'relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
model.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam', metrics= ['acc'] )
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


# 로스: 2.661983013153076
# acc: 0.6856634020805359