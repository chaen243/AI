import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x = [1,2,3]
y = [1,2,3]


w = tf.Variable(111, dtype=tf.float32) #weight
b = tf.Variable(0, dtype=tf.float32) #bias
'''weight, bias 지정 해줘야 사용가능'''

#2. 모델구성
##### y = wx + b ##### 행열 연산에서는 앞뒤 순서가 바뀌면 값이 완전히 다 틀어짐
hypothesis = w * x + b 


#3. 컴파일, 훈련

loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #learning_rate!
train = optimizer.minimize(loss) #(로스를 줄인다)
#model.compile(loss= 'mse', optimizer = 'sgd')


#3-2. 훈련

#sess = tf.compat.v1.Session()

#with가 알아서 session을 close 해줌!
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) #위의 변수를 초기화!

    #model.fit
    epochs= 2000 #for문!
    for step in range(epochs):
        sess.run(train)
        if step % 20 == 0: #tf2의 verbose
            print(step+1, sess.run(loss), sess.run(w), sess.run(b)) #verbose와 model.weight에서 봤던것들
#    sess.close()
