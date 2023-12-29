import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier

#1.데이터
datasets = load_breast_cancer()


x = datasets.data
y= datasets.target

# print(np.unique(y, return_counts= True)) #(array([0, 1]), array([212, 357])
# print(pd.Series(y).value_counts()) #다 똑가틈
# print(pd.DataFrame(y).value_counts()) #다 똑가틈 (dataframe을 쓰면 메모리를 더 씀,행렬데이터는 dataframe만 가능)
# print(pd.value_counts(y)) #다 똑가틈


# ####넘파이####    
# x = np.delete(x, [4,9,-5,-1], axis=1)   
# print(x)
# print(datasets.feature_names)   
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
    
####판다스로 바꿔서 컬럼 삭제 ########   
x = pd.DataFrame(x, columns = datasets.feature_names)
x = x.drop(['mean area', 'mean symmetry','worst compactness','worst fractal dimension' ], axis=1)
#print(x)

#zero_num = len(y[np.where(y==0)]) #넘
#one_num = len(y[np.where(y==1)]) #파
#print(f"0: {zero_num}, 1: {one_num}") #이
#print(df_y.value_counts()) 0 =  212, 1 = 357 #pandas
#print(, unique)
# print("1", counts)
#sigmoid함수- 모든 예측 값을 0~1로 한정시킴.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.68, shuffle= False, random_state= 334)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)

#2. 모델구성
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models:
    model.fit(x_train, y_train)
    results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_predict, y_test)
    print(type(model).__name__, "acc :", acc)
    print(type(model).__name__, ':', model.feature_importances_)
        







#loss : [0.12679751217365265, .acc 0.9707602262496948]

#mms = MinMaxScaler()
#ACC :  0.9890710382513661
#로스 :  [0.047952909022569656

#mms = StandardScaler()
#ACC :  0.9617486338797814
#로스 :  [0.08647095412015915

#mms = MaxAbsScaler()
#ACC :  1.0
#로스 :  [0.05968305468559265

#mms = RobustScaler()
#ACC :  0.9836065573770492
#로스 :  [0.07321718335151672,


#LinearSVC
# model.sore: 0.9617486338797814
# acc : 0.9617486338797814

# DecisionTreeClassifier acc : 0.8743169398907104
# DecisionTreeClassifier : [0.         0.03478658 0.         0.         0.01607915 0.02253338
#  0.         0.         0.         0.00845964 0.00324181 0.
#  0.         0.         0.00275314 0.         0.         0.
#  0.         0.         0.02335996 0.00258756 0.71692205 0.00970337
#  0.06644442 0.         0.07280151 0.01018282 0.         0.01014462]
# RandomForestClassifier acc : 0.9672131147540983
# RandomForestClassifier : [0.05254952 0.01330487 0.04105    0.07485901 0.00872065 0.00753846
#  0.05727807 0.13165028 0.00331791 0.0042324  0.00939497 0.0032734
#  0.00759812 0.03644384 0.00560741 0.00365043 0.0076336  0.00208988
#  0.00659568 0.00476697 0.0993854  0.02165213 0.12562537 0.07781659
#  0.01976186 0.01296562 0.05911353 0.08718944 0.01030168 0.0046329 ]
# GradientBoostingClassifier acc : 0.9617486338797814
# GradientBoostingClassifier : [1.11081732e-03 7.20791541e-03 4.27269102e-04 2.27232667e-03
#  6.02198749e-03 1.23091347e-03 1.28730546e-03 2.05181002e-01
#  5.67953548e-03 1.93534010e-04 6.66489180e-03 2.75549096e-04
#  9.74984421e-04 8.52693459e-03 2.56668217e-04 7.14799229e-04
#  5.67885282e-04 8.38909274e-04 5.22238377e-04 4.04831416e-04
#  2.21797779e-02 5.00401217e-02 4.62374839e-01 9.61220311e-02
#  3.25853114e-02 1.17924481e-03 3.89112865e-02 4.29278565e-02
#  3.12495207e-03 1.94280387e-04]
# XGBClassifier acc : 0.9453551912568307
# XGBClassifier : [0.02393833 0.02867567 0.01606421 0.00278238 0.01022092 0.02669617
#  0.00245873 0.16521733 0.00094606 0.0016372  0.01868628 0.00889118
#  0.01493881 0.01243768 0.00181433 0.00563054 0.00128863 0.00072563
#  0.00243493 0.0019352  0.02740289 0.01897488 0.47150862 0.03872457
#  0.01696069 0.         0.02445868 0.04084015 0.00490873 0.0088006 ]




#컬럼삭제
# DecisionTreeClassifier acc : 0.8306010928961749
# DecisionTreeClassifier : [0.         0.01537984 0.         0.01607915 0.         0.
#  0.         0.00845964 0.         0.         0.         0.0102498
#  0.00976113 0.         0.         0.         0.01014462 0.
#  0.02335996 0.02229753 0.70991406 0.0446355  0.06644442 0.05309154
#  0.01018282 0.        ]
# RandomForestClassifier acc : 0.9672131147540983
# RandomForestClassifier : [0.0409705  0.01612285 0.04939522 0.00848128 0.01475226 0.06491759
#  0.0992564  0.00426419 0.02462824 0.00401765 0.01535982 0.03894243
#  0.005595   0.00693792 0.00563847 0.00633753 0.00506888 0.00651814
#  0.10632258 0.02225495 0.16183736 0.12650491 0.01881434 0.04805105
#  0.08964315 0.0093673 ]
# GradientBoostingClassifier acc : 0.9562841530054644
# GradientBoostingClassifier : [5.99542725e-04 1.29120830e-02 8.88271194e-04 5.88548528e-03
#  1.47962749e-03 1.88689167e-03 2.16112693e-01 2.46416511e-05
#  9.87099602e-03 7.66987405e-05 9.12254635e-04 5.49033444e-03
#  7.07693024e-04 9.85605190e-04 1.49506004e-03 2.85082500e-03
#  1.23221564e-03 1.73705238e-03 1.74280403e-02 5.20325798e-02
#  4.61962524e-01 9.04181320e-02 3.15136182e-02 3.83439768e-02
#  3.95012846e-02 3.65187284e-03]
# XGBClassifier acc : 0.9453551912568307
# XGBClassifier : [0.02469637 0.03156906 0.03284257 0.01212316 0.05130072 0.00073163
#  0.16946085 0.00168904 0.01942759 0.01137466 0.01166372 0.01531015
#  0.00119272 0.00540258 0.00132944 0.00112268 0.00201073 0.00208962
#  0.03876895 0.0186758  0.4205093  0.03138301 0.01864707 0.02739913
#  0.04274113 0.00653829]
