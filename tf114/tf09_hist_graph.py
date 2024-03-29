#실습
#lr 수정해서 epoch 101번 이하
#step = 100, w= 1.99, b= 0.99

import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_data = [1, 2, 3, 4, 5]
y_data = [3, 5, 7, 9, 11]
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


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823) #learning_rate! #0.0823
train = optimizer.minimize(loss) #(로스를 줄인다)
#model.compile(loss= 'mse', optimizer = 'sgd')


#3-2. 훈련

#sess = tf.compat.v1.Session()

#with가 알아서 session을 close 해줌!
'''
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
'''



'''선생님 version'''
loss_val_list = []
w_val_list = []


with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) #위의 변수를 초기화!


    #model.fit
    epochs= 101 #for문!
    for step in range(epochs):
        _,loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict= {x:x_data, y:y_data})

        
        if step % 1 == 0: #tf2의 verbose
            print(step+1, loss_val, w_val, b_val) #verbose와 model.weight에서 봤던것들
            
        loss_val_list.append(loss_val)    
        w_val_list.append(w_val)

    print('='*100)
    #4. 예측
    ################실습########################
    x_pred_data = [6,7,8]
    # 예측값 뽑기
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])



    ###################파이썬방식######################
    # pred = x_pred_data * w_val + b_val
    # print('[6,7,8]의 예측 : ', pred )

    #######placeholder#########
    y_pred2 = x_test * w_val + b_val
    print('[6,7,8]의 예측 : ', sess.run(y_pred2, feed_dict= {x_test: x_pred_data}))
print(loss_val_list)
print(w_val_list)

# plt.plot(loss_val_list)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

# plt.plot(w_val_list)
# plt.xlabel('epochs')
# plt.ylabel('weights')
# plt.show()

# plt.scatter(w_val_list, loss_val_list)
# plt.xlabel('weights')
# plt.ylabel('loss')
# plt.show()


plt.figure(figsize= (15, 15))
plt.subplot(2, 2, 1)  # (행의 수, 열의 수, 위치)
plt.plot(loss_val_list, label='Loss' )
plt.xlabel('Epochs')
plt.ylabel('Loss')

# 2x2 격자의 두 번째 위치에 Weights 대 Epochs 그래프 그리기
plt.subplot(2, 2, 2)
plt.plot(w_val_list, label='Weights')
plt.xlabel('Epochs')
plt.ylabel('Weights')

# 2x2 격자의 세 번째 위치에 Loss 대 Weights 산점도 그리기
plt.subplot(2, 2, 3)
plt.scatter(w_val_list, loss_val_list, label='Loss vs. Weights', color='green')
plt.xlabel('Weights')
plt.ylabel('Loss')

# 2x2 격자의 네 번째 위치는 비워둠

plt.show()
