from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV,HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler
import sklearn as sk
import time

#1. 데이터

# x,y = load_iris(return_X_y= True)
x,y = load_digits(return_X_y= True)


x_train, x_test, y_train , y_test = train_test_split(x, y, shuffle= True, random_state=123, train_size=0.8,stratify= y)
print(x_train.shape)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

print(sk.__version__)
parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]},    # 12
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001,0.0001]},       # 6
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                      # 24
     "gamma":[0.01,0.001, 0.0001], "degree":[3, 4]}                     
]# kernel별로 훈련해서 횟수를 줄여줌.

#2. 모델

#model = SVC(C=1, kernel='linear', degree=3)

model = GridSearchCV(SVC(), 
                     parameters, 
                     cv=kfold, 
                     verbose=1, 
                     refit= True, #디폴트 트루~
                     
                     )#n_jobs=-1) #CPU 다 쓴다!     
# RF = RandomForestClassifier()
print(sk.__version__)
print("==========하빙랜덤서치 1.1.3에서는 돌아감========")
model = HalvingRandomSearchCV(SVC(), 
                     parameters, 
                     cv=5, 
                     verbose=1, 
                     random_state=66,
                     factor=3,
                     min_resources=150,
                     refit= True, #디폴트 트루~
                     n_jobs=-1) #CPU 다 쓴다!

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = accuracy_score(y_test, best_predict)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")

print(sk.__version__)
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

#==========하빙그리드서치========
# n_iterations: 2
# n_required_iterations: 2
# n_possible_iterations: 2
# min_resources_: 100
# max_resources_: 1437
# aggressive_elimination: False
# factor: 4                     #분할한 수
# ----------
# iter: 0
# n_candidates: 14
# n_resources: 100              #cv x 2 x 라벨의 갯수
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# ----------



# n_iterations: 2               #두번돌았다
# n_required_iterations: 2      #두번 도는게 좋다
# n_possible_iterations: 2      #두번 돌수 있다
# min_resources_: 100
# max_resources_: 1437          #train의 최대 수량
# aggressive_elimination: False
# factor: 4
# ----------
# iter: 0
# n_candidates: 14
# n_resources: 100
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# ----------