import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
from sklearn.svm import LinearSVR
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.15, random_state= 3884 ) #282

print(x)
print(y)
print(x.shape, y.shape)

print(datasets.feature_names)
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)

#2. 모델구성

# model = LinearSVR() model.score : 0.11218897208375478 
# model= SVR() model.score : -0.029826507678620162
# model = LinearRegression() model.score : 0.6090435344806548
# model = KNeighborsRegressor() model.score : 0.16267536921893588
# model =DecisionTreeRegressor() model.score : 0.6365768718546047
# model = RandomForestRegressor() model.score : 0.8114871148656582
 

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

#4. 평가, 예측
loss = model.score(x_test, y_test)
print("model.score :", loss)
y_predict = model.predict(x_test) 
result = model.predict(x)



from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)
print("걸린 시간 :", round(end_time - start_time, 2), "초")



#로스 : 0.5067346692085266
#R2 스코어 : 0.6165943551579782
#epochs 5000, batch_size= 200
#8, 16, 10, 8, 4, 1 랜덤 282


#LinearSVR
# model.score : 0.4207089223795233
# r2 : 0.4207089223795233
# 걸린 시간 : 0.39 초