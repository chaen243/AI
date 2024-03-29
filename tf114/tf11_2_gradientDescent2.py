import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

x_train = [1,2,3]
y_train = [1,2,3]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)


w = tf.compat.v1.Variable([10], dtype=tf.float32, name= 'weight')
b = tf.compat.v1.Variable([1], dtype=tf.float32, name= 'bias')


#2.모델
hypothesis = x * w + b


#3-1. 컴파일 // model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse



##################optimizer#############
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823) #learning_rate! #0.0823
#train = optimizer.minimize(loss) #(로스를 줄인다)


#^^^^^  위와같은 내용!

lr = 0.1

gradient = tf.reduce_mean((x * w + b - y) * x)
#loss를 x에 대해서 편미분 한 후 n개로 나눔.


descent = w - lr * gradient
update = w.assign(descent)
##################optimizer#############

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


w_history = []
loss_history = []

for step in range(21):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x: x_train, y: y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)

sess.close()


# with tf.compat.v1.Session() as sess:
#     for i in range(-30, 50):
#         curr_w = i
#         curr_loss = sess.run(loss, feed_dict={w:curr_w})
        
#         w_history.append(curr_w)
#         loss_history.append(curr_loss)
        
# print("=========================W_history====================")  
# print(w_history)
# print("=========================loss_history====================")  
# print(loss_history)      

plt.plot(w_history, loss_history)
plt.xlabel('Weights')
plt.ylabel('Loss')
plt.show()


# plt.plot([-30,-20,-10,1,10,20,30,40,50],[2402.5,1102.5,302.5,0,202.5,902.5,2102.5,3802.5,6002.5])
# plt.xlabel('Weights')
# plt.ylabel('loss')
# plt.show()        
        

