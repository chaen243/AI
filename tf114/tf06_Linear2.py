import tensorflow as tf
tf.compat.v1.set_random_seed(123)

#1. 데이터
x = [1,2,3,4,5]
y = [1,2,3,4,5]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2. 모델구성
hypothesis = w * x + b

#3. 컴파일, 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2000
for step in range(epochs):
    sess.run(train)
    if step % 10 == 0:
        print(step+1, sess.run(loss), sess.run(w), sess.run(b))
sess.close()