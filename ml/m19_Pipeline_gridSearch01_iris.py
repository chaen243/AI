# iter_2 이상 돌리기
# min_resourse 조절
# factor 조절

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split,GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline # **= Pipeline()
import time
import numpy as np

#1. 데이터

x,y = load_iris(return_X_y= True)


x_train, x_test, y_train , y_test = train_test_split(x, y, shuffle= True, random_state=123, train_size=0.8,stratify= y)

print (np.min(x_train), np.max(x_train)) #0.1 7.9
print (np.min(x_test), np.max(x_test)) #0.2 7.7

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

parameters = [
    {"RF__n_estimators": [100, 200], "RF__max_depth": [6, 10, 12], "RF__min_samples_leaf": [3, 10]},
    {"RF__max_depth": [6, 8, 10, 12], "RF__min_samples_leaf": [3, 5, 7, 10]},
    {"RF__min_samples_leaf": [3, 5, 7, 10], "RF__min_samples_split": [2, 3, 5, 10]},
    {"RF__min_samples_split": [2, 3, 5, 10]},
]    
     


#2. 모델구성
     
#model = RandomForestClassifier()


#model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([('mm',MinMaxScaler()),('RF',RandomForestClassifier())])
 
#model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
#model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

#전처리, 모델구성을 한번에 할 수 있음.
#시스템상에서만 바뀌는것, 프린트 했을때는 바뀐값을 출력해주지 않음.




#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측
from sklearn.metrics import accuracy_score


results = model.score(x_test,y_test)
print('model.score', results)

y_predict = model.predict(x_test)

acc= accuracy_score(y_predict,y_test)
print(acc)

# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 200}
# best_score : 0.9583333333333334
# score : 0.9333333333333333
# accuracy_score : 0.9333333333333333
# 최적튠 ACC : 0.9333333333333333
# 걸린시간 : 2.19 초


#randomizer
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=10, min_samples_split=3, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 3, 'min_samples_leaf': 10}
# best_score : 0.9583333333333334
# score : 0.9666666666666667
# accuracy_score : 0.9666666666666667
# 최적튠 ACC : 0.9666666666666667
# 걸린시간 : 1.44 초

#harving
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 3}
# best_score : 0.9666666666666668
# score : 0.9666666666666667
# accuracy_score : 0.9666666666666667
# 최적튠 ACC : 0.9666666666666667
# 걸린시간 : 1.78 초