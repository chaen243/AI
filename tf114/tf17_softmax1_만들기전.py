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
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y)) #mse


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.03)
train = optimizer.minimize(loss)

#위 아래 둘다 가능!
#train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)

#[실습]
from sklearn.metrics import r2_score

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train],
                           feed_dict = {x : x_data, y : y_data})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)
        
y_predict = sess.run(hypothesis, feed_dict= {x : x_data})   
r2 = r2_score(y_data, y_predict) 

print('y_pred :', y_predict)
print('r2 :', r2)    
        