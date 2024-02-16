#Principal Component Analysis 주성분 분석! = 대부분 차원축소의 용도로 많이 사용
#성능향상은 많이 기대할 수 없음. (0이 많은 데이터 같은 경우, 이상치가 많은 경우에는 득이 될 수 있음. )
#지도학습 (y가 있을때)/ 비지도학습 (y가 없음)

#스케일링, PCA후 train_test_split
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn as sk
print(sk.__version__) #1.1.3

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target'] #컬럼명
print(x.shape, y.shape) #(150, 4) (150,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=25)   # 컬럼을 몇개로 줄일것인가(숫자가 작을수록 데이터 손실이 많아짐) 
                            #사용전에 스케일링(주로 standard)을 해서 모양을 어느정도 맞춰준 뒤 사용
x = pca.fit_transform(x)
#print(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle= True)

#2. 모델
model = RandomForestRegressor(random_state=777)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('===============')
print(x.shape)
print('model.score :',results)

evr = pca.explained_variance_ratio_ #설명할수있는 변화율
#n_component 선 갯수에 변화율

print(evr)
print(sum(evr))

evr_cunsum = np.cumsum(evr)
print(evr_cunsum)

import matplotlib.pyplot as plt

plt.plot(evr_cunsum)
plt.grid()
plt.show()
