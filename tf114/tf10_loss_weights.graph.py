import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

# x = [1,2,3]
# y = [1,2,3]

####### 손 계산#######
x = [1,2]
y = [1,2]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})
        
        w_history.append(curr_w)
        loss_history.append(curr_loss)
        
print("=========================W_history====================")  
print(w_history)
print("=========================loss_history====================")  
print(loss_history)      

# plt.plot(w_history, loss_history)
# plt.xlabel('Weights')
# plt.ylabel('Loss')
# plt.show()


plt.plot([-30,-20,-10,1,10,20,30,40,50],[2402.5,1102.5,302.5,0,202.5,902.5,2102.5,3802.5,6002.5])
plt.xlabel('Weights')
plt.ylabel('loss')
plt.show()        
        

