#m31_1에서 뽑은 4가지 결과로 4가지 모델 만들기!
#시간.성능 체크
#결과 예시
# 결과1 PCA=154
# 걸린시간 000초
# acc= 0.000




#scaler
# print(np.argmax(evr_cunsum >= 1.0) +1) #1
# print(np.argmax(evr_cunsum >= 0.999) +1) #683
# print(np.argmax(evr_cunsum >= 0.99) +1) #544
# print(np.argmax(evr_cunsum >= 0.95) +1) #332
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

(x_train, y_train), (x_test,y_test) = mnist.load_data()# '_'자리에는 데이터를 받지 않음.
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)
#x = np.append(x_train, x_test, axis=0)
x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

#print(x.shape) #(70000, 28, 28)
#print(y.shape) #(70000,)
x = x.reshape(70000, 28*28) #(70000, 784)
y = pd.get_dummies(y)

#print(x.shape) # #(70000, 784)

shape = [683, 544, 332, 1]

for i in shape:
    
    scaler = StandardScaler()
    x_1 = scaler.fit_transform(x)
    pca = PCA(i)
    x_1 = pca.fit_transform(x)  

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, stratify= y)

    model = Sequential()
    model.add(Dense(512, input_shape=(x_train[1].shape), activation= 'relu'))
    model.add(Dense(256,activation= 'relu'))
    model.add(Dense(128,activation= 'relu'))
    model.add(Dense(64,activation= 'relu'))
    model.add(Dense(32,activation= 'relu'))
    model.add(Dense(10,activation= 'softmax'))

    #3. 컴파일, 훈련
    model.compile( loss= 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
    start_time = time.time()
    model.fit( x_train, y_train, batch_size=500, verbose=1, epochs= 200, validation_split=0.2, )
    end_time = time.time()

    #4. 평가, 예측
    results = model.evaluate(x_test,y_test)
    print('loss:', results[0])
    print('acc:',  results[1])
    print('PCA',i)
    print('걸린시간: ', round(end_time - start_time,2), '초')
    

# acc: 0.9804571270942688
# PCA 683
# 걸린시간:  14.72 초


# acc: 0.9799428582191467
# PCA 544
# 걸린시간:  27.81 초



# acc: 0.9801714420318604
# PCA 332
# 걸린시간:  27.23 초



# acc: 0.9812571406364441
# PCA 1
# 걸린시간:  28.4 초



# #4. 평가, 예측
# results = model.evaluate(x_test,y_test)
# print('loss:', results[0])
# print('acc:',  results[1])
'''

pca = PCA(n_components=784)   # 컬럼을 몇개로 줄일것인가(숫자가 작을수록 데이터 손실이 많아짐) 
                            #사용전에 스케일링(주로 standard)을 해서 모양을 어느정도 맞춰준 뒤 사용
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ #설명할수있는 변화율
#n_component 선 갯수에 변화율

#print(evr)
#print(sum(evr))

evr_cunsum = np.cumsum(evr)
print(evr_cunsum)

#pca를 통해 0.95 이상인 n_components는 몇개?
#0.95이상
#0.99이상
#0.999이상
#1.0이몇개??

#힌트 np.argmax

print(np.argmax(evr_cunsum >= 1.0) +1) #1
print(np.argmax(evr_cunsum >= 0.999) +1) #683
print(np.argmax(evr_cunsum >= 0.99) +1) #544
print(np.argmax(evr_cunsum >= 0.95) +1) #332

#print(len(evr)) #784



'''


