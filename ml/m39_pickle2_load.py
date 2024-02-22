from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import pickle

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
    


# parameters = {
#     'n_estimators' : 100,
#     'learning_rate' : 0.1,
#     'max_depth' : 3,
#     'min_child_weight' : 10
# }


# model = XGBClassifier()
# path = "C://_data//_save//_pickle_test//"
# model = pickle.load(open(path + 'm39_pickle1_save.dat', 'rb'))

path = "C:\\_data\\_save\\_kaggle_Obesity_Risk\\"
model = pickle.load(open(path + 'submit_19day1745_acc_0.9200385356454721.dat', 'rb'))

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train, 
          eval_set=[(x_test, y_test)],
          verbose= 1, #2부터는 step으로 보여줌

                    
          )
end_time = time.time()


#4. 평가, 예측
  
results = model.score(x_test, y_test)
print('최종점수 :', results)

import datetime
dt = datetime.datetime.now()

import pickle

path = "C:\\_data\\_save\\_kaggle_Obesity_Risk\\"
pickle.dump(model, open(path + f'submit_{dt.day}day{dt.hour:2}{dt.minute:2}_acc_.dat', 'wb'))
