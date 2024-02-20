import os
import tqdm
import json
import random
import pathlib
import imageio
import itertools
import collections
import time


import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow_docs.vis import embed

import keras
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.tools import export_saved_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
# @title Helper functions for loading data and visualizing

NUM_THREADS = 30  # 사용하고 싶은 CPU 쓰레드 수

tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

def split_class_lists(files_for_class, count):
  """
    Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Return:
      split_files: Files belonging to the subset of data.
      remainder: Dictionary of the remainder of files that need to be downloaded.
  """
  split_files = []
  remainder = {}
  for cls in files_for_class:
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder


def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame


class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label.

      Args:
        path: Video file paths.
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.mp4'))
    classes = [p.parent.name for p in video_paths]
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames)
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label
      
      
from pathlib import Path 
download_dir = pathlib.Path("C:/prodownload/")

subset_paths = {}
subset_paths['train'] = Path(os.path.join(download_dir,"train"))
#subset_paths['val'] = Path(os.path.join(download_dir,"val"))
subset_paths['test'] = Path(os.path.join(download_dir,"test"))

# print(subset_paths['train'])

def format_frames(frame, output_size):
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (172,172), frame_step = 2):
   # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))
  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
  need_length = 1 + (n_frames - 1) * frame_step
  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]
  return result

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.mp4'))
    classes = [p.parent.name for p in video_paths]
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()
    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames)
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label
fg = FrameGenerator(subset_paths['train'], 15, training=True)

frames, label = next(fg())

print(f"Shape: {frames.shape}")
print(f"Label: {label}")
train_path = pathlib.Path("C:\prodownload\\train")
test_path = pathlib.Path("C:\prodownload\\test")
val_ds = pathlib.Path("C:\prodownload\\val")
subset_paths = { 'train' :  train_path,
                'test' :  test_path,
                'val' : val_ds
                }

batch_size = 7
num_frames = 150
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], num_frames, training = False),
                                          output_signature = output_signature)
train_ds = train_ds.batch(batch_size)

# val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], num_frames, training = False),
#                                           output_signature = output_signature)
# val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], num_frames),
                                         output_signature = output_signature)
test_ds = test_ds.batch(batch_size)




for frames, labels in test_ds.take(10):
  print("labels : ",labels)
import os  
train_CLASSES = sorted(os.listdir("C:\\prodownload\\train"))
test_CLASSES = sorted(os.listdir("C:\\prodownload\\test"))
# val_CLASSES = sorted(os.listdir("C:\\prodownload\\test"))



train_class_mapping = {i: class_name for i, class_name in enumerate(train_CLASSES)}
test_class_mapping = {i: class_name for i, class_name in enumerate(test_CLASSES)}

print(test_class_mapping)

 
model_id = 'a0'
resolution = 172

tf.keras.backend.clear_session()

backbone = movinet.Movinet(model_id=model_id)
backbone.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([None, None, None, None, 3])

# Create your example input here.
# Refer to the paper for recommended input shapes.
inputs = tf.ones([1, 15, 172, 172, 3])

# [Optional] Build the model and load a pretrained checkpoint.
model.build(inputs.shape)

checkpoint_dir = 'C:\Study\movinet_a0_base'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()




def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  return model


model = build_classifier(batch_size, num_frames, resolution, backbone, 419 )

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

model.summary()


start = time.time()

results = model.fit(train_ds,
#                    validation_data=val_ds,
                    epochs=30,
#                   validation_freq=1,
                    verbose=1,)
                    #callbacks=[cp_callback])
end = time.time()


res = model.evaluate(test_ds, return_dict=True)
print("결과 !!!!!!!!!!!!!!!!!!!!")
print(res)
print("결과 !!!!!!!!!!!!!!!!!!!!")

print("가중치 저장시작!!!!")
model.save("C:\_data\_save\movinet_a0_model_base_30.h5")  
model.save_weights("C:\_data\_save\movinet_a0_model_weights_base_30.h5")
print("가중치 저장완료!!!!")

def get_actual_predicted_labels(dataset):
  
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted


    
fg = FrameGenerator(subset_paths['train'], num_frames, training = True)
label_names = list(fg.class_ids_for_name.keys())

actual, predicted = get_actual_predicted_labels(test_ds)

def return_real_word(y_pred, map):      # 폴더 순서대로 분류돼있던 클래스를  디렉토리 이름으로 변환해주는 함수
    temp = []
    for v in y_pred:
        temp.append(map[v])
    return np.array(temp)
predict_num = predicted.numpy()
actual_num = actual.numpy()

y_pred = return_real_word(predict_num, train_class_mapping)
y_test = return_real_word(actual_num, test_class_mapping)


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

import json
with open('C:\\prodownload\\token_dic.json', 'r') as json_file:          ### 딕셔너리 불러오기
    token_dic = json.load(json_file)
flipped_dict = {v: k for k, v in token_dic.items()}

y_pred = return_real_word(y_pred, flipped_dict)
y_test = return_real_word(y_test, flipped_dict)


print("실제 데이터 : ", y_test) #영상표출 텍스트 표출
print("예측 데이터 : ", y_pred) #텍스트 표출
print("acc : ", acc)







'''
class_to_file_paths = {}
for class_name in subset_paths['train']:
    file_paths = get_files_and_class_names(subset_paths['train'][class_name])  # 각 클래스 디렉토리 내의 모든 파일 경로를 가져오는 함수
    class_to_file_paths[class_name] = file_paths


from IPython.display import HTML
from base64 import b64encode

def display_video(video_path):
    mp4 = open(video_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
    <video width=400 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)

'''



