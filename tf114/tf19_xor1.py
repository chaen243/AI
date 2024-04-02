import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
w = tf.compat.v1.Variable(tf.random_normal([2,1], name = 'weight')) 
b = tf.compat.v1.Variable(tf.zeros([1], name = 'bias')) 

#[실습]

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w)+b)

#3-1. 컴파일

loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1 -y)*tf.log(1-hypothesis))


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.005)
train = optimizer.minimize(loss)

#3-2. 훈련

loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 100001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
                            feed_dict = {x : x_data, y : y_data})
    if step % 20 ==0:
        print(step, "loss:", cost_val, '\t', w_val)
        
        loss_history.append(cost_val)
        
        
#4. 평가, 예측        

# y_pred = x * w_val + b_val #안먹힘!

###y_predict랑 같음
x_test = tf.compat.v1.placeholder(tf.float32, shape= [None, 2])
y_pred = tf.sigmoid(tf.matmul(x_test, w_val) + b_val)
y_predd = sess.run(tf.cast(y_pred>0.5, dtype= tf.float32) , feed_dict= {x_test : x_data})
#tf.cast (반올림 함수) 0~1사이 값을 반올림 처리 해주는것.

print(w_val)
print(type(w_val)) #<class 'numpy.ndarray'> #tensorflow의 데이터 형태!
print('y_pred :', y_predd) 

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, y_predd)
print('acc :', acc)
