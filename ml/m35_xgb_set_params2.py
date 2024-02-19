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

x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=777, test_size= 0.2)#                                                    stratify=y


scaler = StandardScaler()
#scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

split = 3
kfold = KFold(n_splits=split, shuffle= True, random_state=123)
#kfold = KFold(n_splits=split, shuffle= True, random_state=123)


#2. 모델구성
    


parameters = {
    'n_estimators' : 100,
    'learning_rate' : 0.1,
    'max_depth' : 3
}



model = XGBRegressor(random_state = 777, **parameters)

model.set_params(learning_rate=0.01, n_estimators=400,
                 max_depth=7, 
                 reg_alpha=0, 
                 reg_lamdba=1,
                 min_child_weight = 10)

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측
  
print('사용 파라미터 :',model.get_params())
results = model.score(x_test, y_test)
print('최종점수 :', results)

