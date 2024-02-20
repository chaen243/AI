import os
import tqdm
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
import matplotlib
import matplotlib.pyplot as plt
from tensorflow_docs.vis import embed

import keras
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.tools import export_saved_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

NUM_THREADS = 30  # 사용하고 싶은 CPU 쓰레드 수

tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

# @title Helper functions for loading data and visualizing


def frames_from_video_file(video_path, n_frames, output_size = (172,172), frame_step = 15):
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

def to_gif(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('./animation.gif', converted_images, fps=10)
  return embed.embed_file('./animation.gif')
      
from pathlib import Path

download_dir = pathlib.Path("C:/prodownload")

subset_paths = {}
subset_paths['train'] = Path(os.path.join(download_dir,"train"))
subset_paths['val'] = Path(os.path.join(download_dir,"val"))
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

batch_size = 1
num_frames = 100

CLASSES = sorted(os.listdir("C:/prodownload\\train"))

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], num_frames, training = True),
                                          output_signature = output_signature)
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], num_frames),
                                          output_signature = output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], num_frames),
                                         output_signature = output_signature)
test_ds = test_ds.batch(batch_size)

# print(CLASSES)

for frames, labels in train_ds.take(1):
  print(f"Shape: {frames.shape}")
  print(f"Label: {labels.shape}")
  
  
model_id = 'a0'
use_positional_encoding = model_id in {'a3', 'a4', 'a5'}
resolution = 172


backbone = movinet.Movinet(
    model_id=model_id,
    causal=True,
    conv_type='2plus1d',
    se_type='2plus3d',
    activation='hard_swish',
    gating_activation='hard_sigmoid',
    use_positional_encoding=use_positional_encoding,
    use_external_states=False,
)

model = movinet_model.MovinetClassifier(
    backbone,
    num_classes=600,
    output_states=True)

# Create your example input here.
# Refer to the paper for recommended input shapes.
inputs = tf.ones([1, 15, 172, 172, 3])

# [Optional] Build the model and load a pretrained checkpoint.
model.build(inputs.shape)

checkpoint_dir = 'C:\Study\movinet_a0_stream'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()




num_classes = len(CLASSES)
print("num_classes : ", num_classes)
# Construct loss, optimizer and compile the model
#with distribution_strategy.scope():


import tensorflow as tf
from keras import layers, models

def BackboneAndClassifierModel(model_id='a0', num_classes=419,
                               frames_number=15, batch_size=1,
                               resolution=172, dropout=0.05,
                               train_whole_model=False, conv_type='3d', se_type='3d',
                               activation='swish', gating_activation='sigmoid', stream_mode=True,
                               load_pretrained_weights=True, training=True):
    # MoViNet 백본 설정
    # 참고: 여기서 사용된 movinet 및 movinet_model은 실제 코드에서 해당 모델을 불러오는 방식에 따라 달라질 수 있습니다.
    backbone = movinet.Movinet(
        model_id=model_id,
        causal=True,
        conv_type=conv_type,
        se_type=se_type,
        activation= 'relu',
        gating_activation=gating_activation,
        use_external_states=stream_mode
    )

    # 백본 모델의 훈련 가능 상태 설정
    backbone.trainable = False

    # MoViNet 분류기 모델 구성
    inputs = tf.keras.Input(shape=(frames_number, resolution, resolution, 3))
    x = backbone(inputs)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(dropout)(x)

    # 커스텀 분류 레이어 추가
    x = layers.Dense(1280, activation='relu')(x)
    x = layers.Dense(860, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    if load_pretrained_weights:
        # 사전 훈련된 가중치 로드
        checkpoint_dir = 'C:\Study\movinet_a0_stream'  # 실제 경로로 변경 필요
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(checkpoint_path)
        status.assert_existing_objects_matched()

    return model



model = BackboneAndClassifierModel()




loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.005)
model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

checkpoint_path = "C:/study/MiniProject/trained_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 monitor= 'val_accuracy',
                                                 verbose=1)
start = time.time()


results = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=3,
                    validation_freq=1,
                    verbose=1,
                    callbacks=[cp_callback])

end = time.time()

model.save_weights("../movinet_a3_a_weights.h5")
model.save("../movinet_a3_a_model.h5")


result = model.evaluate(test_ds)


print("loss", result[0])
print("acc", result[1])
print('걸린시간 :' , end - start, "초" )


def get_actual_predicted_labels(dataset):
  """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted

def plot_confusion_matrix(actual, predicted, labels, ds_type):
  cm = tf.math.confusion_matrix(actual, predicted)
  ax = sns.heatmap(cm, annot=True, fmt='g')
  sns.set(rc={'figure.figsize':(6, 16)})
  sns.set(font_scale=1.4)
  ax.set_title('Confusion matrix of action recognition for ' + ds_type)
  ax.set_xlabel('Predicted Action')
  ax.set_ylabel('Actual Action')
  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)
  plt.show()
  return cm

fg = FrameGenerator(subset_paths['train'], num_frames, training = True)
label_names = list(fg.class_ids_for_name.keys())

actual, predicted = get_actual_predicted_labels(test_ds)
cm = plot_confusion_matrix(actual, predicted, label_names, 'test')