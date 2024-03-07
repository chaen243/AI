import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import time
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

#1. 데이터
path = "C:\\_data\\kaggle\\bike\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
print(submission_csv)

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count']

print(train_csv.info())
print(test_csv.info())


x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)
y = train_csv['count']
print(y)

print(train_csv.index)
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV, RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수
#kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


#2. 모델구성
Params = {'n_estimators' : 1500,
          'learning_rate' : 0.05,
          'max_depth' : 10,
          'gamma' : 0,
          'min_child_weight' : 0,
          'subsample' : 0.4,
          'colsample_bytree' : 0.8,
          'colsample_bynode' : 1,
          'reg_alpha' : 0,
          'random_state' : 1234,
          }


model = XGBRegressor(tree_method = 'hist', device = "cuda")
model.set_params( **Params)


start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()



print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("r2_score :", r2_score(y_test, y_predict))


print("걸린시간 :", round(end_time - start_time, 2), "초")


for _ in range(len(x_train[0])):
    # 피처 중요도 계산
    feature_importances = model.feature_importances_
    
    # 가장 중요도가 낮은 피처의 인덱스 제거
    least_important_feature_index = np.argmin(feature_importances)
    x_train = np.delete(x_train, least_important_feature_index, axis=1)
    x_test = np.delete(x_test, least_important_feature_index, axis=1)
    
    # 모델 재학습 및 성능 평가
    if x_train.shape[1] > 0:
        model.fit(x_train, y_train, 
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = 0,
          eval_metric = 'rmse')
      
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(f"남은피처 갯수: {x_train.shape[1]}, {r2_score(y_test, y_predict)}")
    else:
        print("No features left.")
        break


