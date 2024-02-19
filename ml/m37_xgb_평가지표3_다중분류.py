from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time

#1. 데이터

# x, y = load_breast_cancer(return_X_y=True)

x, y = load_digits(return_X_y=True)

# x, y = load_diabetes(return_X_y=True)

print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=777, test_size= 0.2, stratify=y)


scaler = StandardScaler()
#scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
    


parameters = {
    'n_estimators' : 100,
    'learning_rate' : 0.1,
    'max_depth' : 3,
    'min_child_weight' : 10
}


model = XGBClassifier()
#model = XGBRegressor(random_state = 777)
#                     , **parameters)

model.set_params(early_stopping_rounds= 50,
    **parameters)
#    early_stopping_rounds =10,
#     n_estimators=4000,
#     learning_rate=0.005, 
#                  min_child_weight = 10)

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train, 
          eval_set=[(x_test, y_test)],
          verbose= 1, #2부터는 step으로 보여줌
        
          # eval_metric = 'rmse' #디폴트
          #eval_metric = 'mae' 
          
          # eval_metric = 'logloss' #이진분류의기본값 # ='binary:logistic' (제일나음)
          #   eval_metric = 'error' #이진분류
          #   eval_metric = 'merror' #다중분류          
          #eval_metric = 'mlogloss' #다중분류의기본값
          
        #   eval_metric = 'auc' #이진 다중 둘다 가능       
                 
                    
          )
end_time = time.time()


#4. 평가, 예측
  
print('사용 파라미터 :',model.get_params())
results = model.score(x_test, y_test)
print('최종점수 :', results) #회귀 디폴트 rmse




# from sklearn.metrics import accuracy_score, r2_score
# from sklearn.metrics import roc_auc_score, f1_score
# y_pred = model.predict(x_test)
# print('r2 :', r2_score(y_test, y_pred))
# print('acc :', accuracy_score(y_test, y_pred))
# print('f1 :', f1_score(y_test, y_pred))
# print('auc :', roc_auc_score(y_test, y_pred))
