import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score



#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


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


#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()



#4. 결과
print('score :', model.score(x_test, y_test))
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2_score :", r2)
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
        print(f"남은피처 갯수: {x_train.shape[1]}, {r2}")
    else:
        print("No features left.")
        break

# r2_score : 0.5625057639285831
# 남은피처 갯수: 7, 0.5625057639285831
# 남은피처 갯수: 6, 0.5625057639285831
# 남은피처 갯수: 5, 0.5625057639285831
# 남은피처 갯수: 4, 0.5625057639285831
# 남은피처 갯수: 3, 0.5625057639285831
# 남은피처 갯수: 2, 0.5625057639285831
# 남은피처 갯수: 1, 0.5625057639285831