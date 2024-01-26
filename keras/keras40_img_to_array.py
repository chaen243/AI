import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print("텐서플로 버전 :", tf.__version__)     #텐서플로 버전 : 2.9.0
print("파이썬 버전 :", sys.version)          #파이썬 버전 : 3.9.18

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img #이미지 가져옴.
from tensorflow.keras.preprocessing.image import img_to_array #이미지를 수치화함.

path = 'c:\_data\image\catdog\Train\Cat\\1.jpg'
img = load_img(path,
                 target_size= (150,150) 
                 )
#타겟사이즈 설정 안하면 원본 사이즈로 바로 나옴.

print(img)
#<PIL.Image.Image image mode=RGB size=150x150 at 0x16E62EABB50>
print(type(img)) #<class 'PIL.Image.Image'>
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape) #(150, 150, 3) -> (281, 300, 3)
print(type(arr)) #<class 'numpy.ndarray'>

#차원증가
img = np.expand_dims(arr, axis=0) #axis 0은 제일 앞 1은 1,2번째 숫자사이 ~~~
print(img.shape) #(1, 150, 150, 3)