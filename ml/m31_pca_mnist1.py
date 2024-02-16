from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

(x_train, _), (x_test,_) = mnist.load_data()# '_'자리에는 데이터를 받지 않음.
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

#x = np.append(x_train, x_test, axis=0)
x = np.concatenate([x_train, x_test], axis=0)
print(x.shape) #(70000, 28, 28)
x = x.reshape(70000, 28*28) #(70000, 784)
print(x.shape)
#pca를 통해 0.95 이상인 n_components는 몇개?
#0.95이상
#0.99이상
#0.999이상
#1.0이몇개??

#힌트 np.argmax
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

pca = PCA(n_components=784)   # 컬럼을 몇개로 줄일것인가(숫자가 작을수록 데이터 손실이 많아짐) 
                            #사용전에 스케일링(주로 standard)을 해서 모양을 어느정도 맞춰준 뒤 사용
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ #설명할수있는 변화율
#n_component 선 갯수에 변화율

#print(evr)
#print(sum(evr))

cunsum = np.cumsum(evr)
print(cunsum)

#pca를 통해 0.95 이상인 n_components는 몇개?
#0.95이상
#0.99이상
#0.999이상
#1.0이몇개??

#힌트 np.argmax
print(np.argmax(cunsum >= 0.95) +1) #154
print(np.argmax(cunsum >= 0.99) +1) #331
print(np.argmax(cunsum >= 0.999) +1) #486
print(np.argmax(cunsum >= 1.0) +1) #713

#print(len(evr)) #784




