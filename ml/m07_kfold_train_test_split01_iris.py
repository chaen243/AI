import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터

x,y = load_iris(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


n_splits=5
#kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)





#2. 모델구성
model = SVC()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))

#4. 예측
y_predict = cross_val_predict(model, x_test, y_test, cv= kfold)
print(y_predict)
print(y_test)

acc= accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :', acc)

