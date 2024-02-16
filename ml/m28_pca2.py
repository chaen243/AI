#Principal Component Analysis 주성분 분석! = 대부분 차원축소의 용도로 많이 사용
#성능향상은 많이 기대할 수 없음. (0이 많은 데이터 같은 경우, 이상치가 많은 경우에는 득이 될 수 있음. )
#지도학습 (y가 있을때)/ 비지도학습 (y가 없음)


#스케일링, PCA전 train_test_split

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
print(sk.__version__) #1.1.3

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target'] #컬럼명
print(x.shape, y.shape) #(150, 4) (150,)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# pca = PCA(n_components=3)   # 컬럼을 몇개로 줄일것인가(숫자가 작을수록 데이터 손실이 많아짐) 
#                             #사용전에 스케일링(주로 standard)을 해서 모양을 어느정도 맞춰준 뒤 사용
# x = pca.fit_transform(x)
#print(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle= True, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


pca = PCA(n_components=1)  
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)





#2. 모델
model = RandomForestClassifier(random_state=777)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('===============')
print(x_train.shape)
print('model.score :',results)


