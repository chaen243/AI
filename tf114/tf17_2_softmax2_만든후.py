import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

#1. 데이터
x_data= [[1,2,1,1],
         [2,1,3,2],
         [3,1,3,4],
         [4,1,5,5],
         [1,7,5,5],
         [1,2,5,6],
         [1,6,6,6],
         [1,7,6,7]]


#원핫 한 상태로만 받아들임
y_data= [[0,0,1], #2
         [0,0,1],
         [0,0,1],
         [0,1,0], #1
         [0,1,0],
         [0,1,0],
         [1,0,0], #0
         [1,0,0]]

x = tf.compat.v1.placeholder(tf.float32, shape= [None, 4])
w = tf.compat.v1.Variable(tf.random_normal([4,3]), name = 'weight') #행렬 연산에 맞추기 위해 4,3이 필요.
b = tf.compat.v1.Variable(tf.zeros([1, 3]), name = 'bias') #shape 잘 볼것!
y = tf.compat.v1.placeholder(tf.float32, shape= [None, 3])

#2. 모델
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)
#E의 x제곱을 n개로 나눈것.

#3-1. 컴파일
#loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y)) #mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
#categorical crossentropy


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

#위 아래 둘다 가능!
#train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)

#[실습]
from sklearn.metrics import r2_score, accuracy_score

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 20001
for step in range(epochs):
    cost_val, _ , w_val= sess.run([loss, train, w],
                           feed_dict = {x : x_data, y : y_data})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)
   
   
#4. 평가. 예측        
y_predict = sess.run(hypothesis, feed_dict= {x : x_data})   
r2 = r2_score(y_data, y_predict) 

print('y_pred :', y_predict)
print('r2 :', r2)    
print(w_val)

#tensorflow 형태
# y_predict = sess.run(tf.argmax(y_predict, 1))
# print(y_predict) #[2 2 2 1 1 1 0 0]
        
#numpy
y_predict = np.argmax(y_predict,1)
print(y_predict)        

y_data = np.argmax(y_data, 1)
# print(y_data)

acc = accuracy_score(y_predict, y_data)
print('acc : ', acc)