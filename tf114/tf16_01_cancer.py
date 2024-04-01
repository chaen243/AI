from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape) #(569, 30)
print(y.shape) #(569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle= True, stratify= y, random_state=213)

# scaler = StandardScaler()
# scaler.fit_transform(x_train)
# scaler.fit_transform(x_test)


y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

x = tf.compat.v1.placeholder(tf.float32, shape= [None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape= [None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1], name= 'weights')) 
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias'))

#2. 모델

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w)+b)

#3-1. 컴파일

loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1 - y)*tf.log(1-hypothesis))

optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.6)
train = optimizer.minimize(loss)

#3-2. 훈련


loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
                                         feed_dict = {x : x_train, y : y_train})
    if step % 20 ==0:
        print(step, 'loss:', cost_val, '\t', w_val)
        
        loss_history.append(cost_val)
        
#4. 평가, 예측
from sklearn.metrics import accuracy_score

y_pred = sess.run(hypothesis, feed_dict= {x:x_test, y: y_test})
y_predict = sess.run(tf.cast(y_pred>0.5, dtype= tf.float32))
acc = accuracy_score(y_test, y_predict)


print('y_pred :', y_predict) 
print('acc :', acc)
       
        
        