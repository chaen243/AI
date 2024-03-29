import tensorflow as tf

tf.compat.v1.set_random_seed(337)

#1. 데이터

x_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]

####실습!###

x = tf.compat.v1.placeholder(tf.float32, shape= [None, 3]) # input_shape와 같음!!!
y = tf.compat.v1.placeholder(tf.float32, shape= [None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1], name = 'weights')) #인풋의 열과 shape를 맞출것!!!
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], name= 'bias')) # *연산이 아니라 + 연산이라 y의 열과 동일한 크기만큼 잡으면 됨!


#2. 모델

# hypothesis = w * w + b
hypothesis = tf.compat.v1.matmul(x,w)+b #행렬연산!!

#3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00005)
train = optimizer.minimize(loss)

#3-2. 훈련
from sklearn.metrics import r2_score

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []

for step in range(2001):
    cost_val, _ , = sess.run([loss, train],
                           feed_dict= {x : x_data, y : y_data})
    if step % 20 ==0:
        print(step, "loss:", cost_val, '\t',)
        
#        w_history.append(w_v)
        loss_history.append(cost_val)
        
#모델         
y_predict = sess.run(hypothesis, feed_dict={x: x_data})
r2 = r2_score(y_data, y_predict)

print('y_pred :', y_predict)
print('r2 :', r2)

sess.close()
        
        