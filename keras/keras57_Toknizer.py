#자연어처리
from keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index) #{'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
# 많이 나오는것/ (같은갯수면) 앞에 있는것 순으로 수치화 해줌.


print(token.word_counts) #([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])

x = token.texts_to_sequences([text])
print(x) #([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])
[[4, 2, 2, 3, 3, 5, 6, 7, 1, 1, 1, 8]]

from keras.utils import to_categorical

x1 = to_categorical(x)
print(x1)
print(x1.shape) #(1, 12, 9)

#1. to_categorical에서 첫번째 0 빼기
x1= x1[:, :, 1:]
print(x1.shape)
print(x1)


# #2. 사이킷런 원핫인코더
# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(sparse=False)
# x2 = ohe.fit_transform(x)
# print(x2)

# #3. 판다스 겟더미
import pandas as pd
import numpy as np

x3 = np.array(x)
print(x3)
x3 = x3.reshape(-1)
x3 = pd.get_dummies(x3)
print(x3.shape)
x3 = x3.values.reshape(1, 12, 8)
print(x3.shape)
