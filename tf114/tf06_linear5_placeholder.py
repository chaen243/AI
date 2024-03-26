import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x = tf.compat.v1.placeholder(tf.float32, shape=[None]) #placeholder로 지정
y = tf.compat.v1.placeholder(tf.float32, shape=[None]) #placeholder로 지정


# w = tf.Variable(111, dtype=tf.float32) #weight
# b = tf.Variable(0, dtype=tf.float32) #bias
'''weight, bias 지정 해줘야 사용가능'''

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #정규분포
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #정규분포



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
    X = [1, 2, 3, 4, 5]
    Y = [3, 5, 7, 9, 11]
    #변수로 미리 지정해주기 위해서는  for문 안의 변수와 다른 이름으로 지정해줘야함

    
    #model.fit
    epochs= 2000 #for문!
    for step in range(epochs):
        y_predict, w_val, b_val, _ = sess.run([loss, w, b, train],
        #feed_dict = {x: [1,2,3,4,5], y : [3,5,7,9,11] }) #값지정해줌
        feed_dict = {x: X, y : Y }) #값지정해줌
        
        if step % 20 == 0: #tf2의 verbose
            print(step+1, y_predict, w_val, b_val) #verbose와 model.weight에서 봤던것들
#    sess.close()
