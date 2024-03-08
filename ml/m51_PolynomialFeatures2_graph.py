import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression #선형회귀
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
plt.rcParams['font.family']= 'Malgun Gothic'

#1. 데이터
np.random.seed(777)
x = 2 * np.random.rand(100 ,1) -1
y = 3 * x**2 + 2 * x + 1 + np.random.randn(100, 1) 
# y = 3x^ + 2x + 1 + 노이즈
# y = 0~1 사이에 노이즈 추가. (전부 양수의 위치에 있게 됨)

#print(x) x = -1 ~ 1 사이

pf = PolynomialFeatures(degree= 2, include_bias= False)
x_poly = pf.fit_transform(x)

print(x_poly)

#2. 모델
model = LinearRegression()
model2 = LinearRegression()

# model = RandomForestRegressor()
# model2 = RandomForestRegressor()

#3. 훈련


model.fit(x, y)
model2.fit(x_poly, y)

#그림 그리기
plt.scatter(x, y, color= 'blue', label = '원데이터')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomal Regression Example')


#다항식 회귀
x_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
x_plot_poly = pf.transform(x_plot)
y_plot = model.predict(x_plot)
y_plot2 = model2.predict(x_plot_poly)
plt.plot(x_plot, y_plot, color = 'red', label = 'Polynomial Regression')
plt.plot(x_plot, y_plot2, color = 'green', label = '그냥')
plt.legend()
plt.show()
