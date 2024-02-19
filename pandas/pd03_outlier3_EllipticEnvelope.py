import numpy as np


aaa = np.array([-10,2,3,4,5,6,700,8,9,10,11,12,50])
print(aaa.shape)

aaa = aaa.reshape(-1,1)
print(aaa.shape) #(13, 1)


from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3) #contaminatiom = 이상치의 비율을 나타냄. 0 ~1사이로 입력

outliers.fit(aaa) #찾기만 하는거기때문에 transform은 ㄴㄴ
result = outliers.predict(aaa)

print(result)