import numpy as np
import pandas as pd


data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan,8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])
#print(data)

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#결측치 확인
print(data.isnull()) #True가 결측치
print(data.isnull().sum())
# x1    1
# x2    2
# x3    0
# x4    3
#print(data.info())
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = SimpleImputer() #디폴트 평균값 strategy = 채우고싶은값

data2 = imputer.fit_transform(data) 
print(data2)

imputer = SimpleImputer(strategy='mean') # 평균

data3 = imputer.fit_transform(data) 
print(data3)


imputer = SimpleImputer(strategy='median') #중위값

data4 = imputer.fit_transform(data) 
print(data4)


imputer = SimpleImputer(strategy='most_frequent') #가장 많은 값

data5 = imputer.fit_transform(data) 
print(data5)


imputer = SimpleImputer(strategy='constant', fill_value= 777) # 상수
data6 = imputer.fit_transform(data) 
print(data6)

imputer = KNNImputer() #knn 알고리즘을 적용해서 predict한 값을 채워줌
data7 = imputer.fit_transform(data)
print(data7)

imputer = IterativeImputer() #선형회귀 (interpolate와 거의 비슷)
data8 = imputer.fit_transform(data)
print(data8)


from impyute.imputation.cs import mice
aaa = mice(data.values, n=10, seed=777)
print(aaa)

print(np.__version__) #1.26.3오류
#1.22.4에서는 잘 돌아감!