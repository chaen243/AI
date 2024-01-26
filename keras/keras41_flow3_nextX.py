from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test)= fashion_mnist.load_data()


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=40,
    zoom_range=0.2,
    fill_mode='wrap', #nearest가 디폴트 #reflet 좌우반전,wrap 감싸줌, 
    shear_range=0.2,
)


augumet_size = 100

#print(x_train[0].shape) #(28, 28)
#plt.imshow(x_train[0])
#plt.show()

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),augumet_size).reshape(-1,28,28,1),  #x
    np.zeros(augumet_size),                                               #np.zeros() = ()안에 모두 0으로 채움. (모양만 맞추는용도) 
    batch_size=augumet_size,                                              
    shuffle=False
    )   #.next()
#==============.next() 쓰지않을때=========
print(x_data)
# print(x_data.shape) #튜플형태라서 에러. flow에서 튜플형태로 반환해줌.
print(x_data[0][0].shape) #(100, 28, 28, 1)
print(x_data[0][1].shape) #(100, )
print(np.unique(x_data[0][1], return_counts= True)) #(array([0.]), array([100], dtype=int64))
print(x_data[0][0][1].shape)
#np.tile(arr,반복갯수)
#np.zero(반복갯수)0을 ()숫자만큼 채워서 모양을 만드는 용도로 씀.




# fig, ax = plt.subplots(nrows=10, ncols=10, figsize= (15, 15))
# x = x_data.next()[0]
# for row in range(10):
#     for col in range(10):
#         ax[row][col].imshow(x[row*col])
#         ax[row][col].axis('off')
        
# plt.show()        
        
plt.figure(figsize=(7, 7))
for i in range(49):             #i를 range만큼 반복한다.
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
  
plt.show()        
        
        
        