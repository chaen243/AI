import os
import tqdm
import random
import pathlib
import imageio
import itertools
import collections

import cv2
import numpy as np
import pandas as pd
import remotezip as rz
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tensorflow_docs.vis import embed

import keras
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

#===========movinet 모델 불러오기===================
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.tools import export_saved_model

def get_class(fname):
  """
    Retrieve the name of the class given a filename.

    Args:
      fname: Name of the file in the UCF101 dataset.

    Return:
      Class that the file belongs to.
  """
  return fname.split('_')[-3]

def get_files_per_class(files):
  """
    Retrieve the files that belong to each class.

    Args:
      files: List of files in the dataset.

    Return:
      Dictionary of class names (key) and files (values).
  """
  files_for_class = collections.defaultdict(list)
  for fname in files:
    class_name = get_class(fname)
    files_for_class[class_name].append(fname)
  return files_for_class

def download_from_zip(zip_url, to_dir, file_names):
  """
    Download the contents of the zip file from the zip URL.

    Args:
      zip_url: Zip URL containing data.
      to_dir: Directory to download data to.
      file_names: Names of files to download.
  """
  with rz.RemoteZip(zip_url) as zip:
    for fn in tqdm.tqdm(file_names):
      class_name = get_class(fn)
      zip.extract(fn, str(to_dir / class_name))
      unzipped_file = to_dir / class_name / fn

      fn = pathlib.Path(fn).parts[-1]
      output_file = to_dir / class_name / fn
      unzipped_file.rename(output_file,)

def split_class_lists(files_for_class, count):

  split_files = []
  remainder = {}
  for cls in files_for_class:
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder



def format_frames(frame, output_size):

  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

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


import shutil

class FrameGenerator:
    def __init__(self, path, n_frames, training=False):
        self.path = pathlib.Path(path)  # pathlib.Path 객체로 변환하여 경로 처리 용이하게 함
        self.n_frames = n_frames
        self.training = training
        # 클래스 이름과 ID 매핑을 생성하는 부분은 유지
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = {name: idx for idx, name in enumerate(self.class_names)}

    def get_files_and_class_names(self):
        video_paths = list(self.path.rglob('*.mp4'))  # 모든 .mp4 파일을 재귀적으로 찾음
        classes = [p.parent.name for p in video_paths]  # 비디오 파일의 부모 디렉토리 이름을 클래스 이름으로 사용
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)  # 훈련 데이터의 경우 무작위로 섞음

        for path, class_name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames)  # 비디오 프레임 추출
            label = self.class_ids_for_name[class_name]  # 클래스 이름에 해당하는 레이블 ID 추출
            yield video_frames, label

def download_hand_sign_local(local_video_dir, num_classes, splits, download_dir):
    # Ensure the download directory exists
    download_dir = pathlib.Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # List directories (classes)
    class_dirs = [d for d in pathlib.Path(local_video_dir).iterdir() if d.is_dir()]
    random.shuffle(class_dirs)  # Shuffle to randomly select classes

    # Select a subset of class directories
    selected_class_dirs = class_dirs[:num_classes]

    # Create splits for each selected class
    dirs = {}
    for split_name, split_percentage in splits.items():
        split_dir = download_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for class_dir in selected_class_dirs:
            # List all video files in the current class directory
            video_files = list(class_dir.rglob('*.mp4'))

            # Calculate the number of files to select for the current split
            num_files = round(len(video_files) * (split_percentage / 100))
            selected_files = random.sample(video_files, num_files)

            # Create a subdirectory for the class in the split directory
            class_split_dir = split_dir / class_dir.name
            class_split_dir.mkdir(parents=True, exist_ok=True)

            # Copy selected files to the class subdirectory in the split directory
            for file_path in selected_files:
                target_path = class_split_dir / file_path.name
                shutil.copy2(file_path, target_path)

            dirs[split_name] = split_dir

    return dirs


local_video_dir = "C:\classified_video_320"
num_classes = 600  # 사용 가능한 클래스 수가 이보다 적을 수 있음
splits = {"train": 60, "val": 20, "test": 20}  # 비율을 백분율로 지정
download_dir = "C:/prodownload"

subset_paths = download_hand_sign_local(local_video_dir, num_classes, splits, download_dir)



     
     
     
     
batch_size = 600
num_frames = 8


# CSV 파일 경로
csv = 'C:\\mini_project\\new_csv_file.csv'

# CSV 파일 읽기
df = pd.read_csv(csv)


labels = df['한국어']


output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int32))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], num_frames, training = True),
                                          output_signature = output_signature)
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], num_frames),
                                          output_signature = output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], num_frames),
                                         output_signature = output_signature)
test_ds = test_ds.batch(batch_size)



for frames, labels in train_ds.take(1):
  print(f"Shape: {frames.shape}")
  print(f"Label: {labels.shape}")
  
  
  
model_id = 'a0'  

tf.keras.backend.clear_session()

resolution = 172

# Create backbone and model.
use_positional_encoding = model_id in {'a3', 'a4', 'a5'}

backbone = movinet.Movinet(
	model_id=model_id,
	causal=True,
	conv_type='2plus1d',
	se_type='2plus3d',
	activation='hard_swish',
	gating_activation='hard_sigmoid',
	use_positional_encoding=use_positional_encoding,
	use_external_states=True,
)

# Create a movinet classifier using this backbone.
model = movinet_model.MovinetClassifier(
	backbone,
	num_classes=600,
	output_states=True)

movinet_hub_url = f'https://tfhub.dev/tensorflow/movinet/{model_id}/stream/kinetics-600/classification/3'

movinet_hub_model = hub.KerasLayer(movinet_hub_url, trainable=True)

pretrained_weights = {w.name: w for w in movinet_hub_model.weights}

model_weights = {w.name: w for w in model.weights}

for name in pretrained_weights:
	model_weights[name].assign(pretrained_weights[name])
 
model = movinet_model.MovinetClassifier(
	backbone=backbone,
	num_classes=len(train_ds.classes),
	output_states=True) 
 
 # Input layer for the frame sequence
image_input = tf.keras.layers.Input(
    shape=[None, None, None, 3],
    dtype=tf.float32,
    name='image')

# Input layers for the different model states.
init_states_fn = movinet_hub_model.resolved_object.signatures['init_states']


state_shapes = {
    name: ([s if s > 0 else None for s in state.shape], state.dtype)
    for name, state in init_states_fn(tf.constant([0, 0, 0, 0, 3])).items()
}

states_input = {
    name: tf.keras.Input(shape[1:], dtype=dtype, name=name)
    for name, (shape, dtype) in state_shapes.items()
}

# Wrap the Movinet model in a Keras model so that it can be finetuned.

inputs = {**states_input, 'image': image_input}

outputs = model(inputs)

# This custom training step ignores the updated states during training as they are only important during inference.
class CustomModel(tf.keras.Model):
	def train_step(self, data):
		# Unpack the data. Its structure depends on your model and
		# on what you pass to `fit()`.
		if len(data) == 3:
			x, y, sample_weight = data
		else:
			sample_weight = None
			x, y = data
		with tf.GradientTape() as tape:
			pred, states = self(x, training=True)  # Forward pass
			
			# Compute the loss value
			# (the loss function is configured in `compile()`)
			loss = self.compiled_loss(y, pred, regularization_losses=self.losses, sample_weight=sample_weight)

		# Compute gradients
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		
		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		
		# Update metrics (includes the metric that tracks the loss)
		self.compiled_metrics.update_state(y, pred, sample_weight=sample_weight)
		
		# Return a dict mapping metric names to current value
		return {m.name: m.result() for m in self.metrics}
      
model = CustomModel(inputs, outputs, name='movinet')


for layer in model.layers[:-1]:
	layer.trainable = False
model.layers[-1].trainable = True

init_states = init_states_fn(tf.shape(tf.ones([batch_size, 8, 172, 172, 3])))

({**init_states, 'image': frame_sequence}, labels)







