import numpy as np


aaa = np.array([[-10,2,3,4,5,
                 6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,
                600,-70000,800,900,1000,210,420,350]])

aaa = aaa.T
print(aaa.shape)



from sklearn.covariance import EllipticEnvelope #두 컬럼을 하나의 데이터로 생각해 사용하려면 컬럼별로 사용하는게 제일 깔끔.
                                                # 성능은 함수로 만든 것과 비슷.
outliers = EllipticEnvelope() #contaminatiom = 이상치의 비율을 나타냄. 0 ~1사이로 입력

outliers.fit(aaa) #찾기만 하는거기때문에 transform은 ㄴㄴ
result = outliers.predict(aaa)

print(result)