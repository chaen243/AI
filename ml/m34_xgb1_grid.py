from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state=1117,
                                                    stratify=y
)

scaler = StandardScaler()
#scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

split = 3
kfold = StratifiedKFold(n_splits=split, shuffle= True, random_state=123)
#kfold = KFold(n_splits=split, shuffle= True, random_state=123)


#2. 모델구성
    

#xgb파라미터
# 'n_estimator : 디폴트 100/ 1~inf/ 정수
'''learning_rate : 디폴트 0.3/ 0~1 /eta''' #적을수록 좋다
#max_depth 디폴트 6 / 0~inf / 정수 #트리의 깊이
# 'gamma' : 디폴트 0/ 0~inf #값이 클수록 데이터 손실이 생김 작을수록 많이 분할되지만 트리가 복잡해짐.
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100]/ 디폴트1/ 0~inf
# 'subsample' : [0, 0.1, 0.3~]/ 디폴트1/ 0~1
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3~~]/ 디폴트1/ 0~1
# 'colsample_bylevel' : [0, 0.1, 0.5]/ 디폴트1 / 0~1
# 'colsample_bynode' : [0, 0.1, 0.5]/ 디폴트1 / 0~1
# 'reg_alpha': [0, 0.1, 0.001]/ 디폴트 0 / 0 ~ inf/ L1 절대값 가중치 규제/ alpha #음수쪽 규제
# 'reg_lambda': [0, 0.1, 0.001]/ 디폴트 1 / 0 ~ inf/ L2 제곱 가중치 규제/ lambda 

parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10], "gamma": [0,2,4]},
    {"max_depth": [4, 6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10],"gamma": [0,2,4]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10],"gamma": [0,2,4]},
    {"min_samples_split": [2, 3, 5, 10], "subsample": [0, 0.3, 1],"gamma": [0,2,4]},
    {"reg_alpha": [0.2351, 0.135], "min_samples_split": [2, 3, 5, 10], "max_depth" : [4, 8, 10, 11]}]



model = RandomizedSearchCV(XGBClassifier(),
                              parameters,
                              cv = kfold,
                              verbose = 1,
                               refit= True,
                               random_state= 123,
                               n_jobs= 22)

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측
from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = accuracy_score(y_test, best_predict)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)                #train의 결과이기 때문에 절대적으로 믿을 수 는 없음.

results = model.score(x_test, y_test)

print('score :', results)

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")



