import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]



#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

#model.add(Dense(10, input-dim=2))
w1 = tf.compat.v1.Variable(tf.random_normal([2,10], name = 'weight1')) 
b1 = tf.compat.v1.Variable(tf.zeros([10], name = 'bias1')) 
layer1 = tf.compat.v1.matmul(x, w1) + b1 #(N,10) 


# layer2 : model.add(Dense(9))
# bias는 덧셈. 열의 갯수와 맞추기.
# weight는 윗레이어 아웃풋 열,원하는 아웃풋 수
w2 = tf.compat.v1.Variable(tf.random_normal([10,8], name = 'weight2')) 
b2 = tf.compat.v1.Variable(tf.zeros([8], name = 'bias2')) 
layer2 = tf.compat.v1.matmul(layer1, w2) + b2 #(N,9) 

# layer3 : model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.random_normal([8,6], name = 'weight3')) 
b3 = tf.compat.v1.Variable(tf.zeros([6], name = 'bias3')) 
layer3 = tf.compat.v1.matmul(layer2, w3) + b3 #(N,8) 


# layer4 : model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.random_normal([6,4], name = 'weight4')) 
b4 = tf.compat.v1.Variable(tf.zeros([4], name = 'bias4')) 
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4) #(N,7) 

# output_layer : model.add(Dense(1), activation= 'sigmoid')
w5 = tf.compat.v1.Variable(tf.random_normal([4,1], name = 'weight5')) 
b5 = tf.compat.v1.Variable(tf.zeros([1], name = 'bias5')) 
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5) #(N,1) 




#[실습]
#hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w)+b)

#3-1. 컴파일

loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1 -y)*tf.log(1-hypothesis))
    #binary_crossentropy

optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype= tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype= tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(8001):
        cost_val, _ = sess.run([loss, train], feed_dict= {x: x_data, y: y_data})
        
        if step % 200 == 0:
            print(step, cost_val)
            
    hypo, pred, acc = sess.run([hypothesis, predicted, acc],
                               feed_dict = {x:x_data, y:y_data})
    print('훈련값 :', hypo)
    print('예측값 :', pred)
    print('정확도 :', acc)