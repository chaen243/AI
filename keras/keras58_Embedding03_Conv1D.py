from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Flatten
from sklearn.metrics import r2_score





#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요','글쎄',
    '별로에요', '생각보다 지루해요','연기가 어색해요',
    '재미없어요','너무 재미없다', '참 재밌네요.',
    '상헌이 바보', '반장 잘생겼다','욱이 또 잔다',    
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index) #{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10,
#'번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20,
# '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '상헌이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '욱이': 28, '또': 29, '잔다': 30} #단어사전의 갯수.

x = token.texts_to_sequences(docs)
print(x) # [[2, 3], [1, 4], [1, 5, 6], 
# [7, 8, 9], [10, 11, 12, 13, 14], [15],
# [16], [17, 18], [19, 20],
# [21], [2, 22], [1, 23],
# [24, 25], [26, 27], [28, 29, 30]]
print(type(x)) #<class 'list'>
#x = np.array(x ) #차원이 달라서 에러뜨는거 보여주기위해 씀.
#print(x) 

from keras.utils import pad_sequences
pad_x = pad_sequences(x, padding= 'pre', 
                      maxlen=5,  
                      truncating= 'pre' )
print(pad_x)
print(pad_x.shape) # (15, 5)

x = pad_x
y = labels

#2. 모델구성

model = Sequential()
model.add(Conv1D(1024, kernel_size= 2, input_shape= (5,1), activation= 'relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



#3. 컴파일, 훈련
model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['acc'] )
model.fit(x, y, epochs= 500, batch_size= 1, )

#4. 평가, 예측
result = model.evaluate(x,y,verbose=0)
y_predict = model.predict(x)
y_predict= np.around(y_predict)
print('로스:', result[0])
print('acc:', result[1])



print('y의 예측값:', y_predict)